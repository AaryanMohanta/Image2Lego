# image2lego evaluation summary

3 mesh(es) x widths [24, 32, 48] x layer types ['brick', 'plate'] = 18 run(s), 0 error(s).

Every solid voxel is given a single fixed placeholder colour (no real photo or colors.csv involved) -- this evaluates the geometry -> bricks -> stability pipeline, not colour fidelity.

| mesh | width | layer type | voxels | bricks | bricks/voxel | articulation (before/after) | single-edge | floating dropped | stability feasible | total slack | seconds/stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| box.glb | 24 | brick | 2400 | 162 | 0.068 | 23 / 1 | 1 | 0 | no | 75.1644 | load=0.015s voxelize=0.080s greedy_and_ground=0.030s legolize_repair=2.386s stability=1.644s |
| box.glb | 24 | plate | 5472 | 446 | 0.082 | 30 / 0 | 0 | 0 | yes | 0.0000 | load=0.009s voxelize=0.191s greedy_and_ground=0.209s legolize_repair=6.519s stability=6.910s |
| box.glb | 32 | brick | 4998 | 386 | 0.077 | 8 / 1 | 1 | 0 | no | 590.7333 | load=0.006s voxelize=0.142s greedy_and_ground=0.069s legolize_repair=5.561s stability=4.738s |
| box.glb | 32 | plate | 10668 | 972 | 0.091 | 8 / 1 | 1 | 0 | no | 121.1081 | load=0.005s voxelize=0.194s greedy_and_ground=0.173s legolize_repair=13.262s stability=23.256s |
| box.glb | 48 | brick | 12612 | 928 | 0.074 | 30 / 1 | 1 | 0 | no | 2294.2311 | load=0.007s voxelize=0.354s greedy_and_ground=0.204s legolize_repair=14.418s stability=24.605s |
| box.glb | 48 | plate | 26244 | 2160 | 0.082 | 6 / 1 | 1 | 0 | no | 635.0400 | load=0.013s voxelize=0.583s greedy_and_ground=0.477s legolize_repair=35.272s stability=151.972s |
| cylinder.glb | 24 | brick | 3864 | 303 | 0.078 | 20 / 19 | 19 | 0 | no | 396.2201 | load=0.006s voxelize=0.300s greedy_and_ground=0.049s legolize_repair=4.208s stability=2.480s |
| cylinder.glb | 24 | plate | 9216 | 855 | 0.093 | 10 / 2 | 2 | 0 | no | 29.2811 | load=0.005s voxelize=0.319s greedy_and_ground=0.187s legolize_repair=10.982s stability=32.725s |
| cylinder.glb | 32 | brick | 7360 | 572 | 0.078 | 14 / 6 | 6 | 0 | no | 1794.6928 | load=0.005s voxelize=0.559s greedy_and_ground=0.108s legolize_repair=8.960s stability=7.774s |
| cylinder.glb | 32 | plate | 17168 | 1513 | 0.088 | 45 / 3 | 3 | 0 | no | 457.7075 | load=0.006s voxelize=0.815s greedy_and_ground=0.317s legolize_repair=21.835s stability=49.484s |
| cylinder.glb | 48 | brick | 17112 | 1336 | 0.078 | 180 / 15 | 16 | 8 | no | 8242.5595 | load=0.006s voxelize=1.358s greedy_and_ground=0.297s legolize_repair=21.849s stability=42.146s |
| cylinder.glb | 48 | plate | 40888 | 3596 | 0.088 | 395 / 57 | 59 | 0 | no | 3748.1099 | load=0.005s voxelize=1.369s greedy_and_ground=0.878s legolize_repair=59.713s stability=285.083s |
| sphere.glb | 24 | brick | 3976 | 490 | 0.123 | 58 / 42 | 31 | 2 | no | 3558.4809 | load=0.020s voxelize=0.030s greedy_and_ground=0.092s legolize_repair=5.669s stability=3.256s |
| sphere.glb | 24 | plate | 9608 | 1468 | 0.153 | 113 / 118 | 77 | 2 | no | 2876.6756 | load=0.018s voxelize=0.221s greedy_and_ground=0.309s legolize_repair=15.498s stability=18.782s |
| sphere.glb | 32 | brick | 7344 | 955 | 0.130 | 113 / 75 | 71 | 0 | no | 6726.6178 | load=0.021s voxelize=0.107s greedy_and_ground=0.187s legolize_repair=11.301s stability=14.327s |
| sphere.glb | 32 | plate | 17792 | 2729 | 0.153 | 173 / 151 | 124 | 2 | no | 5795.5839 | load=0.015s voxelize=0.315s greedy_and_ground=0.587s legolize_repair=30.488s stability=110.286s |
| sphere.glb | 48 | brick | 16976 | 2245 | 0.132 | 316 / 155 | 158 | 2 | no | 16434.6352 | load=0.018s voxelize=0.126s greedy_and_ground=0.488s legolize_repair=29.328s stability=68.305s |
| sphere.glb | 48 | plate | 40960 | 6773 | 0.165 | 465 / 346 | 321 | 0 | no | 13556.8608 | load=0.019s voxelize=1.042s greedy_and_ground=1.529s legolize_repair=86.883s stability=592.108s |
