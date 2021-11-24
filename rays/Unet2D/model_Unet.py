import torch
import torch.nn as nn


class UNet(nn.Module):
    '''A simple 3 layer Unet with some residual blocks'''
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

    
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        channels = 16
        
        self.Conv1 = conv_block(ch_in=in_channels,ch_out=channels)
        self.resblocks1 = res_block(ch_in=channels,ch_out=channels)
        self.Conv2 = conv_block(ch_in=channels,ch_out=channels*2)
        self.Conv3 = conv_block(ch_in=channels*2,ch_out=channels*4)
      
        self.resblocks_b1 = res_block(ch_in=channels*4,ch_out=channels*4)
        self.resblocks_b2 = res_block(ch_in=channels*4,ch_out=channels*4)

      
        
        self.Up3 = up_conv(ch_in=channels*4,ch_out=channels*2)
        self.Up_conv3 = conv_block(ch_in=channels*4, ch_out=channels*2)
        
        self.Up2 = up_conv(ch_in=channels*2,ch_out=channels)
        self.Up_conv2 = conv_block(ch_in=channels*2, ch_out=channels)
        self.resblocks2 = res_block(ch_in=channels,ch_out=channels)
        self.Conv_1x1 = nn.Conv2d(channels,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
      
        x = torch.reshape(x,(-1,145,64))
        x = torch.unsqueeze(x,1)
        pad = torch.zeros((x.shape[0],1,15,64)).cuda()
        x = torch.cat([x,pad],2)
        
        x1 = self.Conv1(x)
        x1 = self.resblocks1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        # x3 = self.resblocks_b1(x3)
        x3 = self.resblocks_b2(x3)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d2 = self.resblocks2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.squeeze(d1,1)
        d1 = d1[:,0:145,:]


        return d1

    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
	   
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
           
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,bias=True),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,bias=True)
            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,bias=True),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
            
        )


    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = res+x
        x = self.conv2(x)
        x = res+x
        return x