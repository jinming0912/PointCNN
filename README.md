setting: K= 2 (without x_transform)

layers:

x = 3

xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in

                [(2, 1, -1, 16 * x, []),

                 (2, 2, 384, 32 * x, []),

                 (2, 2, 128, 64 * x, []),

                 (2, 3, 128, 128 * x, [])]]


performance: 68.15%
