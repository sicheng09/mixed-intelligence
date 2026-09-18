"""Checks for the fixed-anchor joint-action policy used by the frozen experiment."""
import torch
from pufferlib.population_adversary import AnchoredPopulationPolicy, anchored_population_ppo_loss


def test_joint_ratio_and_frozen_exploration():
    q = AnchoredPopulationPolicy()
    k, latent, old_lp = q.sample(1000, .35, torch.Generator().manual_seed(9))
    assert len(k.unique()) == 5
    assert (latent - q.anchors[k]).std() > .3
    old_joint = old_lp + torch.distributions.Normal(q.anchors[k], .35).log_prob(latent).sum(-1)
    with torch.no_grad():
        q.logits[3] += .2
    new_joint = q.distribution().log_prob(k) + torch.distributions.Normal(q.anchors[k], .35).log_prob(latent).sum(-1)
    _, ratio = anchored_population_ppo_loss(q, k, old_lp, torch.ones(1000))
    assert torch.allclose(ratio, (new_joint-old_joint).exp(), atol=1e-6)


def test_reward_selects_component_without_moving_anchors():
    q = AnchoredPopulationPolicy()
    anchors = q.anchors.clone()
    component = torch.arange(5).repeat(20)
    lp = q.distribution().log_prob(component).detach()
    advantage = (component == 4).float() - .2
    optimizer = torch.optim.Adam(q.parameters(), lr=.05)
    for _ in range(4):
        loss, _ = anchored_population_ppo_loss(q, component, lp, advantage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert q.distribution().probs[4] > .2
    assert torch.equal(q.anchors, anchors)


if __name__ == '__main__':
    test_joint_ratio_and_frozen_exploration()
    test_reward_selects_component_without_moving_anchors()
    print('2/2 anchored policy tests passed')
