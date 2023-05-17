import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_joint_mask(data, pred_obs):
    # input size is full size T
    # just visualize one image
    obs_T = 20
    # 
    random_show_ith_image = data['batch'][torch.randint(0,data['batch'].shape[0], (1,))[0]]
    mask = data['batch'] == random_show_ith_image
    first_index = torch.nonzero(mask, as_tuple=True)[0][0] # Find the index of the first True
    ego_index = data['av_index'][random_show_ith_image] - first_index
    
    all_in = data['x'][mask].permute(1,0,2) # [T, M, 2] 
    # all_in_valid_mask = 
    all_out = data['y'][mask].permute(1,0,2)# [T, M, 2]
    pred_obs = pred_obs[:,mask,:,:].permute(0,2,1,3) # [K, T, M, 5]
    num_agent = all_in.shape[1]

    # av_inx = data['av_index'][random_show_ith_image]
    # ego_in = gt_in[] # [T, 5]
    # agents_in = agents_in[random_show_ith_image] # [T, M-1, 5]
    # context_img = context_img[random_show_ith_image] # [M, S, P, 4]
    # pred_obs = pred_obs[:,:,random_show_ith_image,:,:] # [K, T, M, 5]
    # ego_out = ego_out[random_show_ith_image] # [T, 5]
    # agents_out = agents_out[random_show_ith_image] # [T, M-1, 5]
    

    # all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    # all_out = torch.cat((ego_out.unsqueeze(1), agents_out), dim=1) # [T_future, M, 5] 
    figure = plt.figure(figsize=(10,10))
    
    valid_input_agent_x = torch.masked_select(all_in[:,:,2], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,3], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    # valid_input_gs_x = torch.masked_select(context_img[0,:,:,0], context_img[0,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    # valid_input_gs_y = torch.masked_select(context_img[0,:,:,1], context_img[0,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    x_min = min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))-10
    x_max = max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))+10
    y_min = min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))-10
    y_max = max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))+10
    # print(x_min, x_max, y_min, y_max)
    
    # cycol = cycle('bgrcmk')
    # color_str = 'bgrcmk'
    # TODO: max 7 agents. change to colormap
    color_list = ['yellow', 'b', 'g', 'tab:pink', 'r', 'c','m','k']

    """" figure 1: the input image  """
    # ax.imshow(z, aspect="auto")
    plt.subplot(1,3,1)
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())
    
    # plot roads
    # num_agent = context_img.shape[0]
    # num_segment = context_img.shape[1]
    # for i in range(num_segment):
    #     j = 0
    #     # for j in range(num_agent):
    #     # plot given observed input 
    #     valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
    #     valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
    #     # plt.plot(
    #     #     valid_x.cpu(),
    #     #     valid_y.cpu(),
    #     #     "--",
    #     #     color="grey",
    #     #     alpha=0.5,
    #     #     linewidth=1,
    #     #     zorder=0,
    #     # )
    #     plt.scatter(
    #         valid_x.cpu(),
    #         valid_y.cpu(),
    #         c = "grey",
    #         s = 0.2,
    #         alpha=1,
    #         zorder=0,)
   
    # plot agent trajectory, except ego (we keep ego as all visible, so not masked)
    for j in range(num_agent):
        # plot given observed input 
        # input_given_mask = tf.bitwise.bitwise_and(agent_valid_mask[j,:], input_visable_mask)
        input_valid_x = torch.masked_select(all_in[:,j,2], all_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_in[:,j,3], all_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        color = color_list[j]

        if input_valid_x.shape[0] != 0:
            plt.scatter(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                c = color,
                s = 5,
                alpha=1)
            plt.scatter(
                input_valid_x[0].cpu(),
                input_valid_y[0].cpu(),
                c = color,
                # s = 10,
                alpha=1)

    """" figure 2: the predicted image """
    plt.subplot(1,3,2)
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())
    # for i in range(num_segment):
    #     j = 0
    #     # for j in range(num_agent):
    #     # plot given observed input 
    #     valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
    #     valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
    #     # plt.plot(
    #     #     valid_x.cpu(),
    #     #     valid_y.cpu(),
    #     #     "--",
    #     #     color="grey",
    #     #     alpha=0.5,
    #     #     linewidth=1,
    #     #     zorder=0,
    #     # )
    #     plt.scatter(
    #         valid_x.cpu(),
    #         valid_y.cpu(),
    #         c = "grey",
    #         s = 0.2,
    #         alpha=1)
    
    
    # plot predicted trajectory for ego agent. (hidden part for the inputs) [K, T, M, 5]
    # pred_agent = pred_obs[:,:,1:,:]
    # global_pred_agent = convert_local_to_global(agents_out[:obs_T,:,2:].cpu().numpy(), agents_out[obs_T:,:,2:].cpu().numpy(), pred_agent.cpu().numpy())
    # global_pred_agent = torch.from_numpy(global_pred_agent).float().to(agents_out.device)
    global_pred_agent = pred_obs

    num_modes = pred_obs.shape[0]
    for k in range(num_modes):
        # j = 0
        for j in range(global_pred_agent.shape[2]):
        # hidden_mask = tf.bitwise.bitwise_and(agent_valid_mask[j,:], 1-input_visable_mask)
        # predicted_hidden_x = tf.boolean_mask(reg[k,j,:,0], hidden_mask)
        # predicted_hidden_y = tf.boolean_mask(reg[k,j,:,1], hidden_mask)
            color = color_list[j]
            plt.plot(
                global_pred_agent[k,:,j,0].cpu(),
                global_pred_agent[k,:,j,1].cpu(),
                "-",
                color=color,
                alpha=1,
                linewidth=0.5,
            )
    
    

    
    """" figure 3: the ground truth image """
    plt.subplot(1,3,3)
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())
    # for i in range(num_segment):
    #     j = 0

    #     valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
    #     valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))

    #     plt.scatter(
    #         valid_x.cpu(),
    #         valid_y.cpu(),
    #         c = "grey",
    #         s = 0.2,
    #         alpha=1)

    # plot the ground truth trajectory
    # hidden_mask = tf.bitwise.bitwise_and(agent_valid_mask[j,:], 1-input_visable_mask)
    # gt_output_valid_x = tf.boolean_mask(input_agent[j,:,0], hidden_mask)
    # gt_output_valid_y = tf.boolean_mask(input_agent[j,:,1], hidden_mask)
    

    for j in range(num_agent):
        # plot given observed input 
        # input_given_mask = tf.bitwise.bitwise_and(agent_valid_mask[j,:], input_visable_mask)
        input_valid_x = torch.masked_select(all_out[:,j,2], all_out[:,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_out[:,j,3], all_out[:,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        color = color_list[j]

        if input_valid_x.shape[0] != 0:
            # plt.plot(
            #     input_valid_x.cpu(),
            #     input_valid_y.cpu(),
            #     ".",
            #     c = color,
            #     s = 5,
            #     alpha=1,
            #     linewidth=1
            # )
            plt.scatter(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                c = color,
                s = 5,
                alpha=1)
            plt.scatter(
                input_valid_x[0].cpu(),
                input_valid_y[0].cpu(),
                c = color,
                # s = 20,  # dafault is 20
                alpha=1)
    
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlabel("Map X")
    # plt.ylabel("Map Y")
    # global cnt
    # plt.savefig("figures/nuscenes/masked_0.8_" + str(cnt)+".jpg", dpi=700)
    # cnt += 1
    
    # Save the plot to a PNG in memory.
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # # Closing the figure prevents it from being displayed directly inside
    # # the notebook.
    plt.close(figure)
    # buf.seek(0)
    return figure