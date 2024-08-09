import torch

hiera = {
    "hiera_middle":{
        "aquatic mammals": [0, 5],
        "fish": [5, 10],
        "flowers": [10, 15],
        "food containers": [15, 20],
        "fruit and vegetables": [20, 25],
        "household electrical devices": [25, 30],
        "household furniture": [30, 35],
        "insects": [35, 40],
        "large carnivores": [40, 45],
        "large man-made outdoor things": [45, 50],
        "large natural outdoor scenes": [50, 55],
        "large omnivores and herbivores": [55, 60],
        "medium-sized mammals": [60, 65],
        "non-insect invertebrates": [65, 70],
        "people": [70, 75],
        "reptiles": [75, 80],
        "small mammals": [80, 85],
        "trees": [85, 90],
        "vehicles 1": [90, 95],
        "vehicles 2": [95, 100]
    },
    "hiera_high":{
        "animals": [0,1,8,11,12,15,16],
        "plant": [2,4,17],
        "man-made indoor": [3,5,6],
        "man-made outdoor":[9],
        "scenes": [10],
        "invertebrates": [7,13],
        "people": [14],
        "vehicles": [18,19]
    }
}


def prepare_target(gt_label):
    b = gt_label.shape
    gt_label_middle = torch.zeros((b), dtype=gt_label.dtype, device=gt_label.device)
    gt_label_high = torch.zeros((b), dtype=gt_label.dtype, device=gt_label.device)
    for index, middle in enumerate(hiera["hiera_middle"].keys()):
        indices = hiera["hiera_middle"][middle]
        for ii in range(indices[0], indices[1]):
            gt_label_middle[gt_label==ii] = index

    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in indices:
            gt_label_high[gt_label_middle==ii] = index

    return gt_label_middle, gt_label_high
    
    
gt_label = torch.tensor([20,40,90,99,78])
gt_label_middle, gt_label_high = prepare_target(gt_label)

print(gt_label)
print(gt_label_middle)
print(gt_label_high)