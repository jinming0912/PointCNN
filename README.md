setting: K= 1/2 (with x_transform)

layers:

x = 3
rate = 2

xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8//rate, 1, -1, 16 * x, []),
                 (12//rate, 2, 384, 32 * x, []),
                 (16//rate, 2, 128, 64 * x, []),
                 (16//rate, 3, 128, 128 * x, [])]]


performance: 90.88%
