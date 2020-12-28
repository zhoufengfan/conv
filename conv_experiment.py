import torch

if __name__ == '__main__':
    t1 = torch.ones((1, 3, 5, 3)).float()
    conv0 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2)
    t2 = conv0(t1)
    print("t2.size() is\t", t2.size())
