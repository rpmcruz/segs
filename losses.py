def dice_score(y_pred, y_true, smooth=1):
    dice = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice

def dice_loss(y_pred, y_true, smooth=1):
    return 1 - dice_score(y_pred, y_true, smooth)
