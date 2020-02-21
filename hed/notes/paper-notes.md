Crux of the idea from : https://arxiv.org/abs/1504.06375

0. Not all edge are created equal : Edges are tied to image scale, Edges at lower scales contribute to discovery of edges at higher scales - fine to coarse in deep network.
1. Layers at different depth in a deep network operate at larger receptive field size (scale)
2. Use fully convolution network to take in arbitrary sizes image and produce feature maps at different conv layer
3. Feed feature map from each conv block to deconv (upsampling to GT size) and minimize against GT with weighted (%age of edge pixels in GT) binary cross entropy loss after sigmoid
4. Take Weighted average of output from each side layer (weights learned through training) and compare with GT with weighted binary cross entropy
5. Fine tune the entire network with combined loss 3. + 4.
6. `[Question]` : Weights of binary cross entropy either local or global since 90% of pixels are non-edge
7. `[Question]` : Change GT to be the consensus of atleast three annotators. Noisy/small scale edge have lower consensus
8. Rotate images to 16 different angles + largest crop to augment data since training sets are small (Authors provided the data-set which already has augmentation applied)
9. No use of Mean-shift to find thinner edge/post processing for now.
10. `[Question]`: Treat edge detection problem as regression or binary classification ?
11. `[Question]`: The authors use different learning for differnent levels in the network side layers. Is that needed ?
12. `[Question]`: Output from layer 5 dont look as good as in the paper?
12. `[Question]`: Rather than using single deconv at each output layer, use stacked deconv with non-linearity?
