import torch
import torch_struct


def torch_struct_matrix_img(output, C=11):
    # Lower diag eye from diagonal
    lower_diag_eye = torch.eye(C).cuda()
    tril_indices = torch.tril_indices(row=C, col=C)
    lower_diag_eye[tril_indices[0], tril_indices[1]] = 1

    # Upper diag eye
    upper_diag_eye = torch.zeros((C, C)).cuda()
    triu_indices = torch.triu_indices(row=C, col=C, offset=1)
    upper_diag_eye[triu_indices[0], triu_indices[1]] = 1

    # negative_infinity_constraint_value
    negative_infinity_constraint_value = -1e2  # -1e2
    expand_v = output.unsqueeze(2).expand(*output.size(), C)
    expand_h = expand_v.transpose(2, 3)

    # Final potential
    potentials = expand_h*lower_diag_eye + \
        negative_infinity_constraint_value * upper_diag_eye

    return potentials


def torch_struct_matrix_flow(output, C=11):
    lower_diag = torch.zeros((C, C)).cuda()
    tril_indices_1 = torch.tril_indices(row=C, col=C, offset=-1)
    lower_diag[tril_indices_1[0], tril_indices_1[1]] = 1

    upper_diag = torch.zeros((C, C)).cuda()
    triu_indices = torch.triu_indices(row=C, col=C, offset=1)
    upper_diag[triu_indices[0], triu_indices[1]] = 1

    eye = torch.eye(C).cuda()

    negative_infinity_constraint_value = -1e2  # -1e2

    no_change_mul = [(torch.matmul(eye.unsqueeze(
        2), out_batch[:, 0].unsqueeze(0)).T) for out_batch in output]
    no_change_mul = torch.stack(no_change_mul, 0)

    change_mul = [(torch.matmul(lower_diag.T.unsqueeze(
        2), out_batch[:, 1].unsqueeze(0)).T) for out_batch in output]
    change_mul = torch.stack(change_mul, 0)

    potential = torch.zeros(
        (output.shape[0], output.shape[1]+1, C, C), dtype=torch.float).cuda()

    potential[:, 1:] = (no_change_mul + change_mul +
                        negative_infinity_constraint_value*upper_diag)

    return potential


def torch_struct_matrix_adjust_size(potentials):
    # adjusted size!
    potentials_smaller_dim = potentials[:, 1:]
    potentials_smaller_dim[:, 0] += potentials[:, 0]
    return potentials_smaller_dim.contiguous()


def linear_chain_distribution(img_pred, flow_pred, label_orig, num_samples,
                              C=11, verbose=False, img=False, flow=False,
                              use_weighting=False, lambda_p=.1):
    log_potentials = 0
    if img:
        log_potentials += torch_struct_matrix_img(img_pred)
    if flow:
        log_potentials += lambda_p*torch_struct_matrix_flow(flow_pred)
    if not flow and not img:
        raise ValueError("Flow and img argument can't be both false!")

    log_potentials = torch_struct_matrix_adjust_size(log_potentials)
    if verbose:
        print(log_potentials.shape)
    if verbose:
        print(img_pred.shape)
    if verbose:
        print(flow_pred.shape)
    if verbose:
        print(label_orig.shape)
    # number of classes
    if verbose:
        print(C)
    # Create batch dimention label
    label = label_orig.view(-1, num_samples)  # .transpose(0, 1)
    if verbose:
        print(label.shape)
    # Create batch dimention potentials
    if verbose:
        print(log_potentials)
    # Compute distribution (each time)
    dist = torch_struct.LinearChainCRF(log_potentials)
    # Compute the argmax through viterby algorithm
    argmax_C_by_C = dist.argmax
    # Parts = a ( batch x 500 x 8 x 8  ) sparse target matrix
    parts_C_by_C = dist.struct.to_parts(label, C).type_as(argmax_C_by_C)
    # Convert to events:
    argmax_labels = dist.from_event(argmax_C_by_C)[0].detach().cpu().numpy()
    parts_labels = dist.from_event(parts_C_by_C)[0].detach().cpu().numpy()

    return argmax_C_by_C, argmax_labels, parts_C_by_C, parts_labels, dist
