import torch
import torch.nn.functional as F

def calculate_derivative(phy_data):
    """
    Calculates the derivative of the physical data tensor for a specified channel.

    Parameters:
    phy_data (torch.Tensor): The physical data tensor with dimensions [batch_size, time, channels, height, width].

    Returns:
    torch.Tensor: The derivative of the physical data for the specified channel.
    """
    # Selecting the specified channel for all frames
    q= phy_data[:, :, 1, :, :]
    #print("Batch1 Shape (q0 to q8):", q.shape)
    
    dt=0.5
    
    # Calculating the difference from q0 to q8 (q1-q0, q2-q1, ..., q8-q7)
    dq = torch.diff(q, dim=1)
    
    #print("Difference Shape (dq1 to dq8):", dq.shape)

    # Compute derivative using finite difference
    dq_dt = dq / dt
    #print("Derivative Shape (dq1 to dq8):", dq_dt.shape)

    return dq_dt

def calculate_spatial_derivative(phy_data, dx=1.0, dy=1.0):
    """
    Calculates the spatial derivative of the physical data tensor along x and y coordinates.

    Parameters:
    q (torch.Tensor): The physical data tensor with dimensions [batch, time, y, x].
    dx (float): The distance between grid points along the x-axis in kilometers.
    dy (float): The distance between grid points along the y-axis in kilometers.

    Returns:
    tuple of torch.Tensor: The derivatives of the physical data along the x and y coordinates.
    """
    q= phy_data[:, :, 1, :, :]
    # Calculate the partial derivative with respect to x (horizontal axis)
    dqdx = torch.diff(q, dim=3) / dx  # axis 3 corresponds to the x-coordinate in (batch, time, y, x)
    
    # Calculate the partial derivative with respect to y (vertical axis)
    dqdy = torch.diff(q, dim=2) / dy  # axis 2 corresponds to the y-coordinate in (batch, time, y, x)
    
    # Replicate the last column for dqdx
    dqdx_padded = torch.cat((dqdx, dqdx[:, :, :, -1:]), dim=3)
    
    # Replicate the last row for dqdy
    dqdy_padded = torch.cat((dqdy, dqdy[:, :, -1:, :]), dim=2)
    
    return dqdx_padded, dqdy_padded


def RQ(phy_data):
    dq_dt=calculate_derivative(phy_data)
    #print(dq_dt.shape)

    dqdx, dqdy = calculate_spatial_derivative(phy_data, dx=1.0, dy=1.0)
    #print("dqdx shape:", dqdx.shape)
    #print("dqdy shape:", dqdy.shape)

    windu10=phy_data[:, :, 4, :, :]
    #print("wind shape:", windu10.shape)
    term2= torch.mul(dqdx,windu10)
    #print("result1:", term2.shape)

    windv10=phy_data[:, :, 5, :, :]
    #print("wind shape:", windv10.shape)
    term3= torch.mul(dqdy,windv10)
    #print("result1:", term3.shape)

    windu100=phy_data[:, :, 2, :, :]
    #print("wind shape:", windu100.shape)
    term4= torch.mul(dqdx,windu100)
    #print("result1:", term4.shape)

    windv100=phy_data[:, :, 3, :, :]
    #print("wind shape:", windv100.shape)
    term5= torch.mul(dqdy,windv100)
    #print("result1:", term5.shape)
    
    neg_ones = -1 * torch.ones(256, 256)

    dq_dt1=dq_dt * neg_ones
    #print(dq_dt1.shape)

    term2_f=term2[:, 1:, :, :] * neg_ones
    #print(term2_f.shape)

    term3_f=term3[:, 1:, :, :] * neg_ones
    #print(term3_f.shape)

    term4_f=term4[:, 1:, :, :]* neg_ones
    #print(term4_f.shape)

    term5_f=term5[:, 1:, :, :]* neg_ones
    #print(term3_f.shape)

    eva=phy_data[:, 1:, 0, :, :]
    #print(eva.shape)
    sum1=torch.add(dq_dt1,term2_f)
    #print(sum1.shape)
    sum2=torch.add(term3_f,term4_f)
    #print(sum2.shape)
    sum3=torch.add(term5_f,eva)
    #print(sum3.shape)

    sum4=torch.add(sum1,sum2)
    #print(sum4.shape)
    final=torch.add(sum3,sum4)
    #print(sum5.shape)
    return final


    
    
    