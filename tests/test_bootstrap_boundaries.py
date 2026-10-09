"""Exercise source-extracted rollout stores and native advantage boundary kernels."""
import ast
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'pufferlib/pufferl.py'
TREE=ast.parse(SOURCE.read_text())
CLASS=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='PuffeRL')

def extracted_store(sync=False):
    method=next(n for n in CLASS.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    if sync:
        loop=next(n for n in method.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='start')
        block=next(n for n in loop.body if isinstance(n,ast.With))
        body=block.body
    else:
        loop=next(n for n in method.body if isinstance(n,ast.While))
        blocks=[n for n in loop.body if isinstance(n,ast.With)]
        block=blocks[-1]
        body=block.body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pending_rows' for t in n.targets))
        body=body[start:start+3]
    fn=ast.FunctionDef(name='store',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ['self','config','env_id','pending_rows','pending_mask','o','o_device','r','d','value']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['store']

@pytest.mark.parametrize('offload',[False,True])
@pytest.mark.parametrize('sync',[False,True])
@pytest.mark.parametrize('dones',[[0,0],[0,1],[1,0],[1,1]])
def test_actual_done_follows_physical_rows(offload,sync,dones):
    class Buffer:pass
    b=Buffer();b.observations=torch.zeros(8,3,1);b.rewards=torch.zeros(8,3);b.values=torch.zeros(8,3)
    b.bootstrap_indices=torch.tensor([6,2],dtype=torch.int32);b.bootstrap_terminals=torch.zeros(8)
    pending=b.bootstrap_indices.long();mask=pending>=0
    o=torch.tensor([[60.],[20.]]);r=torch.tensor([.6,.2]);d=torch.tensor(dones);v=torch.tensor([.6,.2])
    extracted_store(sync)(b,{'cpu_offload':offload,'bptt_horizon':2},slice(0,2),pending,mask,o,o,r,d,v)
    assert b.bootstrap_terminals[6]==dones[0] and b.bootstrap_terminals[2]==dones[1]
    assert b.observations[6,2,0]==60 and b.observations[2,2,0]==20
    assert b.rewards[6,2]==r[0] and b.rewards[2,2]==r[1]
    assert b.bootstrap_terminals[[0,1,3,4,5,7]].sum()==0


def test_terminal_extension_preserves_true_last_flag():
    f=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='extend_terminals')
    ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),str(SOURCE),'exec'),ns)
    old=torch.tensor([[0.,0.],[0.,1.]])
    assert torch.equal(ns['extend_terminals'](old,torch.tensor([1.,0.])),torch.tensor([[0.,0.,1.],[0.,1.,0.]]))
    with pytest.raises(ValueError):ns['extend_terminals'](old,None)


def test_all_update_paths_use_saved_terminal():
    text=SOURCE.read_text()
    assert 'adv_terminal[:, -1] = self.terminals[:, -1]' not in text
    assert 'torch.cat((mb_terminals, mb_terminals[:, -1:])' not in text
    assert text.count('adv_terminal[:, -1] = self.bootstrap_terminals')==2
    assert 'self.bootstrap_terminals[selected_idx]' in text
    assert 'self.bootstrap_terminals[idx]' in text


def test_native_cpu_cuda_boundary_equivalence():
    def extract(path,name):
        text=path.read_text();start=text.index('void '+name+'(');brace=text.index('{',start);depth=1;end=brace+1
        while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
        return text[start:end]
    cpu=extract(ROOT/'pufferlib/extensions/pufferlib.cpp','puff_advantage_row')
    cuda=extract(ROOT/'pufferlib/extensions/cuda/pufferlib.cu','puff_advantage_row_cuda')
    main=r'''int main(){float v[4]={0,0,0,0},r[4]={0,1,100,2},d[4]={0,0,0,0},t[3]={0,1,0},i[3]={1,1,1},a[3]={0,0,0},b[3]={0,0,0};
    puff_advantage_row(v,r,d,t,i,a,.98,.95,1,1,3);puff_advantage_row_cuda(v,r,d,t,i,b,.98,.95,1,1,3);
    for(int j=0;j<3;j++)if(fabs(a[j]-b[j])>1e-6)return 1;
    if(fabs(a[0]-1)>1e-6)return 2;
    float v2[3]={0,0,10},r2[3]={0,0,1},d2[3]={0,0,1},t2[2]={0,0},i2[2]={1,1},a2[2]={0,0};
    puff_advantage_row(v2,r2,d2,t2,i2,a2,.98,.95,1,1,2);if(fabs(a2[1]-1)>1e-6)return 3;
    d2[2]=0;puff_advantage_row(v2,r2,d2,t2,i2,a2,.98,.95,1,1,2);if(fabs(a2[1]-10.8)>1e-5)return 4;
    return 0;}'''
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'test.cpp').write_text('#include <cmath>\n'+cpu+'\n'+cuda+'\n'+main)
        subprocess.run(['g++',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
        subprocess.run([str(p/'test')],check=True)
