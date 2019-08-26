setting: aligned K= 2 12layers (with x_transform + 4 channel-wise)

layers:

x = 3

xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in

                [(2, 1, -1, 16 * x, []),#0

                 (2, 2, 1024, 16 * x, []),#1

                 (2, 2, 768, 32 * x, []),#2

                 (2, 2, 768, 32 * x, []),#3

                 (2, 2, 512, 64 * x, []),#4

                 (2, 2, 512, 64 * x, []),#5

                 (2, 3, 384, 128 * x, []),#6

                 (2, 3, 384, 128 * x, []),#7

                 (2, 3, 256, 256 * x, []),#8

                 (2, 3, 256, 256 * x, []),#9

                 (2, 3, 128, 512 * x, []),#10

                 (2, 3, 128, 512 * x, [])]]#11
                

performance: 88.01%
