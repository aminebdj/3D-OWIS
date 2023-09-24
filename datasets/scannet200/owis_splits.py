
#wall and floor are excluded from the training in all splits


SPLIT_A_SUBSET_1_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 39, 42, 43, 45, 46, 47, 48, 50, 57, 58, 59, 60, 62, 63, 66, 71, 74, 78, 79, 81, 90, 98, 99, 115, 116, 161, 170, 171, 172, 178]
SPLIT_A_SUBSET_2_IDS = [31, 35, 36, 37, 38, 40, 41, 44, 49, 51, 52, 53, 54, 55, 56, 61, 64, 65, 68, 69, 70, 72, 75, 76, 77, 80, 83, 86, 87, 88, 89, 93, 95, 96, 97, 101, 102, 103, 104, 106, 107, 108, 111, 113, 118, 123, 126, 127, 128, 132, 133, 140, 143, 144, 149, 152, 166, 167, 169, 173, 174, 175, 176, 180, 184, 189, 195, 197]
SPLIT_A_SUBSET_3_IDS = [67, 73, 82, 84, 85, 91, 92, 94, 100, 105, 109, 110, 112, 114, 117, 119, 120, 121, 122, 124, 125, 129, 130, 131, 134, 135, 136, 137, 138, 139, 141, 142, 145, 146, 147, 148, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 168, 177, 179, 181, 182, 183, 185, 186, 187, 188, 190, 191, 192, 193, 194, 196]
SPLIT_A_SUBSET_1_LABELS = ['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture', 'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet', 'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity', 'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel', 'door', 'clothes', 'whiteboard', 'bed', 'bathtub', 'desk', 'wardrobe', 'clothes dryer', 'radiator', 'shelf']
SPLIT_A_SUBSET_2_LABELS = ["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"]
SPLIT_A_SUBSET_3_LABELS = ["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", "fire alarm", "power strip", "calendar", "poster", "luggage"]

SPLIT_B_SUBSET_1_IDS = [0, 1, 2, 3, 5, 8, 9, 11, 16, 18, 19, 22, 25, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 42, 44, 47, 51, 55, 56, 62, 64, 65, 72, 75, 76, 77, 79, 80, 81, 87, 88, 94, 96, 101, 102, 112, 113, 118, 121, 127, 136, 139, 140, 141, 143, 144, 145, 151, 157, 164, 165, 170, 172, 173, 177, 186, 187, 188, 189, 194, 195, 196, 197]
SPLIT_B_SUBSET_2_IDS = [4, 6, 7, 12, 14, 15, 17, 20, 37, 41, 43, 48, 52, 54, 57, 60, 63, 67, 68, 71, 74, 78, 83, 84, 85, 86, 89, 91, 93, 95, 100, 105, 106, 108, 111, 120, 124, 125, 130, 132, 133, 134, 142, 148, 155, 161, 166, 167, 175, 178, 179, 181, 184, 190, 192]
SPLIT_B_SUBSET_3_IDS = [10, 13, 21, 23, 24, 28, 31, 34, 40, 45, 46, 49, 50, 53, 58, 59, 61, 66, 69, 70, 73, 82, 90, 92, 97, 98, 99, 103, 104, 107, 109, 110, 114, 115, 116, 117, 119, 122, 123, 126, 128, 129, 131, 135, 137, 138, 146, 147, 149, 150, 152, 153, 154, 156, 158, 159, 160, 162, 163, 168, 169, 171, 174, 176, 180, 182, 183, 185, 191, 193]
SPLIT_B_SUBSET_1_LABELS = ['alarm clock', 'backpack', 'bag', 'bed', 'blanket', 'case of water bottles', 'ceiling', 'closet', 'closet door', 'closet wall', 'clothes', 'coat rack', 'container', 'curtain', 'door', 'dresser', 'dumbbell', 'fan', 'guitar case', 'hat', 'ironing board', 'lamp', 'laptop', 'laundry basket', 'laundry hamper', 'luggage', 'mattress', 'mini fridge', 'nightstand', 'object', 'pillow', 'poster', 'power outlet', 'purse', 'rack', 'recycling bin', 'shelf', 'shoe', 'sign', 'storage bin', 'storage organizer', 'suitcase', 'tissue box', 'wardrobe', 'decoration', 'armchair', 'bench', 'bicycle', 'candle', 'chair', 'coffee table', 'couch', 'dining table', 'end table', 'fireplace', 'jacket', 'keyboard piano', 'light', 'music stand', 'ottoman', 'piano', 'picture', 'pillar', 'plant', 'potted plant', 'rail', 'sofa chair', 'speaker', 'stool', 'table', 'tv', 'tv stand', 'vacuum cleaner']
SPLIT_B_SUBSET_2_LABELS = ['guitar', 'paper towel roll', 'book', 'bookshelf', 'cart', 'furniture', 'blackboard', 'projector', 'seat', 'folded chair', 'office chair', 'projector screen', 'whiteboard', 'bin', 'bucket', 'bulletin board', 'copier', 'machine', 'mailbox', 'paper cutter', 'printer', 'column', 'storage container', 'blinds', 'structure', 'water bottle', 'ball', 'board', 'box', 'cabinet', 'cd case', 'ceiling light', 'clock', 'computer tower', 'cup', 'desk', 'divider', 'file cabinet', 'headphones', 'keyboard', 'monitor', 'mouse', 'paper', 'person', 'power strip', 'radiator', 'stand', 'telephone', 'tray', 'tube', 'window', 'windowsill', 'pipe', 'stair rail', 'stairs']
SPLIT_B_SUBSET_3_LABELS = ['bar', 'basket', 'bathroom cabinet', 'bathroom counter', 'bathroom stall', 'bathroom stall door', 'bathroom vanity', 'bathtub', 'bottle', 'broom', 'clothes dryer', 'cushion', 'doorframe', 'fire alarm', 'hair dryer', 'handicap bar', 'ledge', 'light switch', 'mat', 'mirror', 'paper towel dispenser', 'plunger', 'scale', 'shower', 'shower curtain', 'shower curtain rod', 'shower door', 'shower floor', 'shower head', 'shower wall', 'sink', 'soap dish', 'soap dispenser', 'toilet', 'toilet paper', 'toilet paper dispenser', 'toilet paper holder', 'toilet seat cover dispenser', 'towel', 'trash bin', 'washing machine', 'closet rod', 'dustpan', 'laundry detergent', 'stuffed animal', 'bowl', 'calendar', 'coffee kettle', 'coffee maker', 'counter', 'dish rack', 'dishwasher', 'fire extinguisher', 'kitchen cabinet', 'kitchen counter', 'microwave', 'oven', 'paper bag', 'plate', 'range hood', 'refrigerator', 'stove', 'toaster', 'toaster oven', 'trash can', 'vent', 'water cooler', 'water pitcher', 'crate', 'ladder']

SPLIT_C_SUBSET_1_IDS = [0, 2, 4, 5, 6, 8, 20, 33, 36, 37, 38, 39, 40, 43, 46, 50, 51, 52, 57, 60, 61, 66, 69, 73, 75, 76, 77, 78, 88, 91, 94, 102, 105, 109, 111, 119, 121, 123, 124, 127, 128, 129, 132, 136, 139, 140, 144, 146, 147, 148, 154, 155, 160, 161, 164, 168, 170, 171, 172, 173, 177, 178, 180, 185, 195, 196]
SPLIT_C_SUBSET_2_IDS = [3, 10, 13, 14, 15, 16, 17, 18, 19, 22, 25, 27, 28, 34, 42, 44, 47, 48, 55, 58, 59, 62, 64, 65, 67, 70, 71, 79, 81, 83, 84, 85, 87, 89, 90, 95, 96, 97, 99, 100, 106, 112, 113, 116, 118, 120, 122, 126, 130, 131, 135, 141, 149, 151, 156, 162, 163, 165, 167, 176, 183, 184, 186, 187, 192, 194]
SPLIT_C_SUBSET_3_IDS = [1, 7, 9, 11, 12, 21, 23, 24, 26, 29, 30, 31, 32, 35, 41, 45, 49, 53, 54, 56, 63, 68, 72, 74, 80, 82, 86, 92, 93, 98, 101, 103, 104, 107, 108, 110, 114, 115, 117, 125, 133, 134, 137, 138, 142, 143, 145, 150, 152, 153, 157, 158, 159, 166, 169, 174, 175, 179, 181, 182, 188, 189, 190, 191, 193, 197]
SPLIT_C_SUBSET_1_LABELS = ['basket', 'trash can', 'stair rail', 'toaster oven', 'laundry hamper', 'bulletin board', 'dining table', 'stuffed animal', 'bathroom vanity', 'box', 'ceiling', 'potted plant', 'luggage', 'closet wall', 'paper cutter', 'desk', 'object', 'rail', 'tissue box', 'plate', 'keyboard', 'hat', 'copier', 'shower head', 'bed', 'paper towel dispenser', 'fire extinguisher', 'paper towel roll', 'backpack', 'water bottle', 'bathroom cabinet', 'stove', 'laundry basket', 'alarm clock', 'headphones', 'piano', 'guitar', 'bag', 'door', 'speaker', 'water cooler', 'shoe', 'water pitcher', 'dumbbell', 'furniture', 'decoration', 'radiator', 'plunger', 'shower', 'bar', 'hair dryer', 'suitcase', 'cabinet', 'chair', 'board', 'laundry detergent', 'whiteboard', 'vacuum cleaner', 'power outlet', 'storage bin', 'computer tower', 'mailbox', 'shelf', 'ledge', 'pillar', 'toilet paper']
SPLIT_C_SUBSET_2_LABELS = ['ironing board', 'divider', 'oven', 'dish rack', 'shower door', 'mini fridge', 'bicycle', 'laptop', 'armchair', 'couch', 'coffee kettle', 'counter', 'structure', 'pipe', 'bowl', 'shower curtain rod', 'sofa chair', 'clothes dryer', 'coffee table', 'stairs', 'toilet seat cover dispenser', 'machine', 'paper bag', 'book', 'blinds', 'monitor', 'shower wall', 'curtain', 'closet', 'telephone', 'fan', 'ball', 'bucket', 'sign', 'mirror', 'clock', 'nightstand', 'tv stand', 'handicap bar', 'poster', 'blanket', 'cup', 'recycling bin', 'lamp', 'scale', 'mouse', 'wardrobe', 'ottoman', 'paper', 'power strip', 'fireplace', 'doorframe', 'toilet', 'trash bin', 'case of water bottles', 'light', 'washing machine', 'guitar case', 'sink', 'bathtub', 'ladder', 'bookshelf', 'column', 'clothes', 'keyboard piano', 'music stand']
SPLIT_C_SUBSET_3_LABELS = ['mattress', 'toaster', 'stool', 'plant', 'folded chair', 'microwave', 'cushion', 'bench', 'soap dispenser', 'storage organizer', 'shower curtain', 'cart', 'kitchen counter', 'towel', 'blackboard', 'tv', 'printer', 'stand', 'rack', 'bathroom counter', 'closet rod', 'bottle', 'range hood', 'purse', 'candle', 'person', 'coffee maker', 'light switch', 'storage container', 'bathroom stall door', 'shower floor', 'kitchen cabinet', 'refrigerator', 'fire alarm', 'tube', 'toilet paper holder', 'ceiling light', 'picture', 'end table', 'closet door', 'file cabinet', 'crate', 'toilet paper dispenser', 'pillow', 'mat', 'bathroom stall', 'broom', 'container', 'seat', 'jacket', 'dresser', 'dustpan', 'table', 'projector', 'window', 'windowsill', 'tray', 'cd case', 'soap dish', 'office chair', 'dishwasher', 'vent', 'coat rack', 'calendar', 'bin', 'projector screen']

"""
IDS for currently known, unknown, and previously known classes for each split and task.

IMPORTANT: IDS in this script correspond to the index of the prediction of the class in the classification head, 
           and NOT the label IDS defined by scannet200 benchmark.
"""

# UNKNOWN_CLASSES_IDS: IDS of the unknown classes
UNKNOWN_CLASSES_IDS = {
    'A':{
        'task1': SPLIT_A_SUBSET_2_IDS+SPLIT_A_SUBSET_3_IDS,
        'task2': SPLIT_A_SUBSET_3_IDS,
        'task3': None
    },
    'B':{
        'task1': SPLIT_B_SUBSET_2_IDS+SPLIT_B_SUBSET_3_IDS,
        'task2': SPLIT_B_SUBSET_3_IDS,
        'task3': None
    },
    'C':{
        'task1': SPLIT_C_SUBSET_2_IDS+SPLIT_C_SUBSET_3_IDS,
        'task2': SPLIT_C_SUBSET_3_IDS,
        'task3': None
    }
}

# KNOWN_CLASSES_IDS: IDS of the currently known classes
KNOWN_CLASSES_IDS = {
    'A': {
        'task1': SPLIT_A_SUBSET_1_IDS,
        'task2': SPLIT_A_SUBSET_2_IDS,
        'task3': SPLIT_A_SUBSET_3_IDS
    },
    'B': {
        'task1': SPLIT_B_SUBSET_1_IDS,
        'task2': SPLIT_B_SUBSET_2_IDS,
        'task3': SPLIT_B_SUBSET_3_IDS
    },
    'C': {
        'task1': SPLIT_C_SUBSET_1_IDS,
        'task2': SPLIT_C_SUBSET_2_IDS,
        'task3': SPLIT_C_SUBSET_3_IDS
    }
}

# PREV_KNOWN_CLASSES_IDS: IDS of the previously known classes
PREV_KNOWN_CLASSES_IDS = {
    'A': {
        'task1': None,
        'task2': SPLIT_A_SUBSET_1_IDS,
        'task3': SPLIT_A_SUBSET_1_IDS+SPLIT_A_SUBSET_2_IDS
    },
    'B': {
        'task1': None,  
        'task2': SPLIT_B_SUBSET_1_IDS,
        'task3': SPLIT_B_SUBSET_1_IDS+SPLIT_B_SUBSET_2_IDS
    },
    'C': {
        'task1': None,  
        'task2': SPLIT_C_SUBSET_1_IDS,
        'task3': SPLIT_C_SUBSET_1_IDS+SPLIT_C_SUBSET_2_IDS
    }
}

"""
LABELS for currently known, unknown, and previously known classes for each split and task.
"""

PREV_KNOWN_CLASSES_LABELS = {
    'A': {
        'task1': None,
        'task2': SPLIT_A_SUBSET_1_LABELS,
        'task3': SPLIT_A_SUBSET_1_LABELS+SPLIT_A_SUBSET_2_LABELS
    },
    'B': {
        'task1': None,  
        'task2': SPLIT_B_SUBSET_1_LABELS,
        'task3': SPLIT_B_SUBSET_1_LABELS+SPLIT_B_SUBSET_2_LABELS
    },
    'C': {
        'task1': None,  
        'task2': SPLIT_C_SUBSET_1_LABELS,
        'task3': SPLIT_C_SUBSET_1_LABELS+SPLIT_C_SUBSET_2_LABELS
    }
}

KNOWN_CLASSES_LABELS = {
    'A' :{
        'task1': SPLIT_A_SUBSET_1_LABELS,
        'task2': SPLIT_A_SUBSET_2_LABELS,
        'task3': SPLIT_A_SUBSET_3_LABELS
    },
    'B': {
        'task1': SPLIT_B_SUBSET_1_LABELS,
        'task2': SPLIT_B_SUBSET_2_LABELS,
        'task3': SPLIT_B_SUBSET_3_LABELS
    },
    
    'C': {
        'task1': SPLIT_C_SUBSET_1_LABELS,
        'task2': SPLIT_C_SUBSET_2_LABELS,
        'task3': SPLIT_C_SUBSET_3_LABELS
    }
}

UNKNOWN_CLASSES_LABELS = {
    'A' :{
        'task1': SPLIT_A_SUBSET_2_LABELS+SPLIT_A_SUBSET_3_LABELS,
        'task2': SPLIT_A_SUBSET_3_LABELS,
        'task3': None
    },
    'B': {
        'task1': SPLIT_B_SUBSET_2_LABELS+SPLIT_B_SUBSET_3_LABELS,
        'task2': SPLIT_B_SUBSET_3_LABELS,
        'task3': None,
    },
    'C': {
        'task1': SPLIT_C_SUBSET_2_LABELS+SPLIT_C_SUBSET_3_LABELS,
        'task2': SPLIT_C_SUBSET_3_LABELS,
        'task3': None,
    }
}

