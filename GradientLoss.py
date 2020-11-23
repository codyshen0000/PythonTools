import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns 

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, y):
        # batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        # count_h = self._tensor_size(x[:, :, 1:, :])
        # count_w = self._tensor_size(x[:, :, :, 1:])
        gard_x_h, gard_x_w = x[:, :, 1:, :] - x[:, :, :h_x - 1, :], x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        gard_y_h, gard_y_w = y[:, :, 1:, :] - y[:, :, :h_x - 1, :], y[:, :, :, 1:] - y[:, :, :, :w_x - 1]
        
        loss = torch.nn.functional.l1_loss(gard_x_h, gard_y_h) +\
            torch.nn.functional.l1_loss(gard_x_w, gard_y_w)
        # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * loss
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GradientLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GradientLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def GW_loss(self, x1, x2):
        robert_x = [-1, 0, 1]
        robert_y = [-1, 0, 1]
        # sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
        # sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
        #     sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
        # sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
        # sobel_x = sobel_x.type_as(x1)
        # sobel_y = sobel_y.type_as(x1)
        
        b, c, w, h = x1.shape
        robert_x = torch.FloatTensor(robert_x).reshape(1,1,1,3).expand(c, 1, 1, 3)
        robert_y = torch.FloatTensor(robert_y).reshape(1,1,3,1).expand(c, 1, 3, 1)
    
        weight_x = nn.Parameter(data=robert_x, requires_grad=False).to(self.device)
        weight_y = nn.Parameter(data=robert_y, requires_grad=False).to(self.device)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=(0,1), groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=(0,1), groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=(1,0), groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=(1,0), groups=c)
        # _, (ax1, ax2) = plt.subplots(1,2)
        # # plt.hist(Ix1.detach().cpu().reshape(-1),bins=50, density=True)
        # ax1.imshow(Ix1.detach().cpu().squeeze().permute(1,2,0))
        # # ax1.set_yscale("log")
        # # plt.yscale("log")
        # ax2.imshow(Ix2.detach().cpu().squeeze().permute(1,2,0), cmap='gray')
        # plt.show()
        
        dx = torch.mean(((abs(Ix1 - Ix2)+1e-8)**0.7)**(1./0.7))#torch.norm(torch.abs(Ix1 - Ix2)+1e-8, 0.7).mean()
        dy = torch.mean(((abs(Iy1 - Iy2)+1e-8)**0.7)**(1./0.7))#torch.norm(torch.abs(Iy1 - Iy2)+1e-8, 0.7).mean()
        # dx = torch.pow(torch.abs(Ix1 - Ix2 + 1e-8), 0.7).mean()
        # dy =  torch.pow(torch.abs(Iy1 - Iy2 + 1e-8), 0.7).mean()
       
        loss = (dx + dy).mean() #* torch.abs(x1 - x2).mean() 

        return loss.mean()
    
    def forward(self, x, y):
        return self.GW_loss(x, y)


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    # @staticmethod
    def gradient_map(self, x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        # _, (ax1, ax2) = plt.subplots(1,2)
        histy = (t - b).detach().cpu().reshape(-1).numpy()
        histx = (r - l).detach().cpu().reshape(-1).numpy()
        # sns.distplot(hist, hist=True, bins=255, kde=False,rug=False, norm_hist=True, kde_kws={"color":"seagreen", "lw":1 }, hist_kws={ "color": "b" })
        plt.hist2d(histx, bins=255, density=True, color='blue')
        plt.yscale("log",basey=2)
        # ax1.set_yscale("log",basey=2)
        # ax2.hist((t - b).detach().cpu().reshape(-1),bins=50, density=False)
        # ax2.set_yscale("log",basey=2)
        plt.savefig('Figure_7_grad_y_normal.png')
        assert False
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad

  
def main():
    # x = Variable(torch.FloatTensor([[[1,2],[2,3]],[[1,2],[2,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[3,1],[4,3]],[[3,1],[4,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[1,1,1], [2,2,2],[3,3,3]],[[1,1,1], [2,2,2],[3,3,3]]]).view(1, 2, 3, 3), requires_grad=True)
    import imageio 
    img = imageio.imread('7.png')
    img_ = imageio.imread("7.png")

    img = torch.from_numpy(img).unsqueeze(0).permute([0,3,1,2]).float().cuda()
    img_ = torch.from_numpy(img_).unsqueeze(0).permute([0,3,1,2]).float().cuda()
    addition = GradientPriorLoss()
    z = addition(img, img_)
    # x = torch.FloatTensor([1,2,3,4,5,6,7,8]).view(2, 1, 2, 2)
    # y = torch.ones([2, 1,2,2], requires_grad=True)
    # x.requires_grad= True
    # addition = GradientLoss()
    # z = addition(x,y)
    # print(x,y) 
    # print(z.data) 
    # z.backward()
    # print(x.grad)

if __name__ == '__main__':
    import torch.distributions as dist
    import math 
    x = torch.Tensor([-200.7532,2,221.6192])
    x.requires_grad = True
    y = torch.pow(torch.abs(x), 1)
    y.mean().backward()
    print(x.grad)
    main()
    # def log_prob(value, loc=0, var=1, p=0.7 ):
    #     # loc = torch.Tensor(loc)
    #     # var = torch.Tensor(var)
    #     # return -torch.log(2 * var) - torch.abs(var - loc)**p / var

    #     return -(torch.abs(value - loc) ** p) / var - math.log(2 * var) 
    
    # value = torch.linspace(-100,100,2000)
    # x = torch.distributions.Laplace(0,1)
    # y = torch.distributions.Normal(0,1)
    
    # mix = dist.Categorical(torch.ones(5,))
    # comp = dist.Normal(0, torch.rand(5,))
    # # z = torch.distributions.MixtureSameFamily
    # gmm = dist.MixtureSameFamily(mix, comp)
    # plt.plot(value, torch.exp(x.log_prob(value)),color='blue')
    # plt.plot(value, torch.exp(y.log_prob(value)),color='red')
    # plt.plot(value, torch.exp(gmm.log_prob(value)),color='green')
    # plt.plot(value, torch.exp(log_prob(value)),color='yellow')

    # # plt.ylim(10**0, -10**2)
    # plt.yscale('log')
    
    # # plt.fill_between(value, torch.exp(x.log_prob(value)))
    # plt.show()

    # main()