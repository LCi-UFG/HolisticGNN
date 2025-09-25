import torch
from torch.distributions import (
    Dirichlet, 
    Gamma
    )


class DMPGM:
    def __init__(
        self,
        num_nodes,
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters,
        device=None):

        self.N = num_nodes
        self.K = num_components
        self.alpha = concentration
        self.prior_amp = prior_concentration
        self.mcmc_iters = mcmc_iters
        self.device = device
        self.u = torch.zeros(
            self.N + 1, device=self.device
            )
        self.wk = torch.ones(
            self.K + 1, self.N + 1, 
            device=self.device
            )
        self.pi = torch.ones(
            self.K + 1, device=self.device) / (self.K + 1)
        self.c = None
        self.z = None
        self.hmc_stepsize = 0.1
        self.hmc_leapfrog_steps = 5

    def logp_u(self, u: torch.Tensor, 
            counts_w0: torch.Tensor) -> torch.Tensor:
        w0 = u.exp()
        sw0 = w0.sum()
        log_prior = -sw0
        log_like = (counts_w0 * u).sum() - sw0

        return log_prior + log_like

    def mcmc_update(
        self, 
        edge_index):

        i, j = edge_index.cpu()
        idx = torch.stack([i, j], dim=1) 

        for _ in range(self.mcmc_iters):
            # Compute scores with strong numerical guards
            scores = (
                self.pi.unsqueeze(1)
                * self.wk[:, idx[:,0].to(self.device)]
                * self.wk[:, idx[:,1].to(self.device)]
            )
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
            scores = scores.clamp(min=1e-8, max=1e6)
            self.c = torch.multinomial(
                scores.t(), num_samples=1).squeeze(1)

            # Limit Poisson rate to avoid extreme counts
            lam = scores.sum(dim=0)
            lam = torch.nan_to_num(lam, nan=0.0, posinf=1e6, neginf=0.0)
            lam = lam.clamp(min=1e-8, max=1e6).cpu()
            z_cpu = torch.poisson(lam)
            z_cpu[z_cpu == 0] = 1
            self.z = (idx, z_cpu.to(self.device))

            flat_c = self.c.repeat(2)                  
            flat_nodes = torch.cat(
                [idx[:,0], idx[:,1]], dim=0
                )  
            flat_z = z_cpu.repeat(2).to(self.device)
            counts = torch.zeros(
                self.K + 1, self.N + 1, 
                device=self.device
                )
            counts.index_put_(
                (flat_c, flat_nodes.to(self.device)), 
                flat_z, accumulate=True
                )
            # Build concentration with guards
            u_exp = torch.exp(torch.clamp(self.u, min=-20.0, max=20.0))
            shape = u_exp.unsqueeze(0) + counts
            shape = torch.nan_to_num(shape, nan=1.0, posinf=1e6, neginf=1e-6)
            shape = shape.clamp(min=1e-6, max=1e6)
            self.wk = Gamma(
                shape, torch.ones_like(shape)).sample()

            counts_pi = torch.bincount(self.c, 
                minlength=self.K + 1).float().to(self.device)
            counts_pi[0] += self.alpha 
            self.pi = Dirichlet(
                counts_pi + self.prior_amp).sample()
            self.pi = self.pi / self.pi.sum()
            counts_w0 = counts[0]

            with torch.enable_grad():
                u_old = self.u.clone().detach().requires_grad_(True)
                p_old = torch.randn_like(u_old)
                U = lambda uu: -self.logp_u(uu, counts_w0)
                H_old = U(u_old) + 0.5 * (p_old**2).sum()
                p = p_old - 0.5 * self.hmc_stepsize * torch.autograd.grad(
                    U(u_old), u_old)[0]
                u = u_old.clone()

                for _ in range(self.hmc_leapfrog_steps):
                    u = u + self.hmc_stepsize * p
                    grad_u = torch.autograd.grad(U(u), u)[0]
                    p = p - self.hmc_stepsize * grad_u
                p = p - 0.5 * self.hmc_stepsize * torch.autograd.grad(U(u), u)[0]
                H_new = U(u) + 0.5 * (p**2).sum()
        
                if torch.rand(1, device=self.device
                    ) < torch.exp(H_old - H_new):
                    # Keep u bounded to prevent overflow in exp
                    self.u = torch.clamp(u.detach(), min=-20.0, max=20.0)

    def virtual_graph(self):

        idx, z = self.z 
        idx = idx.to(self.device) 
        z   = z.to(self.device)

        idx2 = torch.cat([idx, idx[:, [1, 0]]], dim=0)
        z2   = torch.cat([z, z], dim=0)
        deg = torch.zeros(self.N, device=self.device)
        deg = deg.index_add(0, idx2[:, 0], z2)    
        inv_sqrt = deg.pow(-0.5)
        inv_sqrt[torch.isinf(inv_sqrt)] = 0.0
        weight = z2 * inv_sqrt[
            idx2[:, 0]] * inv_sqrt[idx2[:, 1]
            ]

        return idx2.t(), weight