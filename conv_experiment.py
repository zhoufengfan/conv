import torch

if __name__ == '__main__':
    img_batch = torch.rand((1, 3, 5, 4)).float()
    conv0 = torch.nn.Conv2d(in_channels=3, out_channels=7, kernel_size=2)
    # PyTorch will initialize the Weight and Bias automatically.
    W, b = list(conv0.parameters())
    # print("W is:\n", W)
    # print("b is:\n", b)
    print("W.shape is:", W.shape)
    print("b.shape is:", b.shape)
    # print("list(conv0.parameters()) is:\n", list(conv0.parameters()))
    t2 = conv0(img_batch)
    print("t2.size() is\t", t2.size())
    # Should take all input channel into consideration.
    print("img_batch.shape is", img_batch.shape)
    print("torch.sum(W[0] * t1[0]) + b[0]", torch.sum(W[0] * img_batch[0][:, :2, :2]) + b[0])
    print("t2[0][0][0][0] is", t2[0][0][0][0])
