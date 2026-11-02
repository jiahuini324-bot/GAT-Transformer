from copy import deepcopy

import torch
from torch_geometric.loader import DataLoader

from model import DecoupledPhysicalMSE


def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    device = next(model.parameters()).device
    total_loss, total_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = criterion(model(batch), batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * batch.num_nodes
        total_nodes += batch.num_nodes
    if total_nodes == 0:
        raise ValueError("Empty dataset")
    return total_loss / total_nodes


@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        loss = criterion(model(batch), batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
    if total_nodes == 0:
        raise ValueError("Empty dataset")
    return total_loss / total_nodes


def fit(
    model,
    training_data,
    validation_data,
    scale=(1.0, 1.0, 1.0),
    epochs=300,
    batch_size=72,
    learning_rate=1e-4,
    minimum_learning_rate=1e-6,
    device=None,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    criterion = DecoupledPhysicalMSE(scale).to(device)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=minimum_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_state, best_loss = [], None, float("inf")
    for epoch in range(epochs):
        training_loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
        validation_loss = evaluate(model, validation_loader, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history
