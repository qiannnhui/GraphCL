import torch
import os

def save_checkpoint(epoch, model, optimizer, best_test_acc, filename):
    """將模型狀態、優化器狀態和當前 epoch/準確度儲存到檔案。"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': best_test_acc,
    }
    torch.save(state, filename)
    print(f"===> Checkpoint saved to {filename} (Epoch {epoch})")


def load_checkpoint(filename, model, optimizer, device):
    """從檔案載入模型狀態、優化器狀態，並返回 epoch 和 best_test_acc。"""
    if os.path.isfile(filename):
        print(f"===> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_test_acc = checkpoint.get('best_test_acc', 0.0) # 容錯處理，如果舊ckpt沒有
        print(f"===> Loaded checkpoint (Epoch {checkpoint['epoch']}, Best Test Acc {best_test_acc:.4f})")
        return start_epoch, best_test_acc
    else:
        print(f"===> No checkpoint found at '{filename}'")
        return 0, 0.0 # 從第 0 epoch 開始