import imageio

images = []

for i in range(100):
    images.append(imageio.imread('../images/' + str(i+1) + '.png'))
imageio.mimsave('./video/attacker_faster_sim1.gif', images)

