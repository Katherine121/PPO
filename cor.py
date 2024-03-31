dis_list = [50, 75, 100, 125, 150]
height_list = [160, 180, 200, 220, 240]
noise_list = [15, 20, 25, 30, 35]
max_ep_len_list = [128, 192, 256, 320, 384]
validate_num_nodes_list = [60, 80, 100, 120, 140]
radius_list = [800, 900, 1000, 1100, 1200]

# the random position deviation
HEIGHT_NOISE = {200: 25}
# the kinds of disturbances
NOISE_DB = [["ori", "ori"], ["ori", "random"],
            # ["cutout", "random"], ["rain", "random"], ["snow", "random"], ["fog", "random"], ["bright", "random"]
            ]
# the random position deviation
# HEIGHT_NOISE = {100: 25, 150: 25, 200: 25, 250: 50, 300: 50}
