import numpy as np
import tensorflow as tf



# Function Factory based on work by: Pi-Yueh Chuang <pychuang@gwu.edu>
# URL: https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993#file-tf_keras_tfp_lbfgs-py


# =========================================================================================
# Function Factory for Physics-Informed Neural Networks:
# =========================================================================================

def PINN_function_factory(model, loss_domain, loss_bc, loss_ic):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss_bc [in]: loss function for the boundary conditions.
        loss_ic [in]: loss function for the initial conditions.
        loss_domain [in]: loss function for the domain points that must satisfy the given PDE.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    # shapes = tf.shape_n(model.trainable_variables)
    # Get layer shapes
    # shapes = tf.shape_n(model.trainable_variables) # This is not working for me. I use the following instead
    shapes = []
    for i in range(len(model.trainable_variables)):
        shapes.append(model.trainable_variables[i].shape)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_d = loss_domain()
            loss_b = loss_bc()
            loss_i = loss_ic()
            loss = loss_d + loss_b +loss_i

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter % 100 == 0:
            #print('Epoch: {}\t Loss = {}\t Loss_domain = {}\t Loss_bc = {}\t Loss_ic = {}'.format(f.iter, loss, loss_d, loss_b, loss_i))
            tf.print("Epoch:", f.iter, "\t Loss = ", loss, '\t Loss_domain = ', loss_d, '\t Loss_bc = ', loss_b, '\t Loss_ic = ', loss_i)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[[loss, loss_d, loss_b, loss_i]], Tout=[])

        return loss, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f



# =========================================================================================
# Function Factory for Inverse Physics-Informed Neural Networks:
# =========================================================================================

def IPINN_function_factory(model, loss_domain, loss_solution, lambda_value):
  
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    
        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss_solution [in]: loss function for the given solution points.
            loss_domain [in]: loss function for the domain points that must satisfy the given PDE.
            lambda_value [in]: external trainable parameter of mdoel 
    
        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = [variable.shape for variable in model.trainable_variables]
        n_tensors = len(shapes)


        # prepare required information for dynamic_stitch and dynamic_partition
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        # Ensure the last element in idx matches the last element in shapes
        idx[-1] = tf.reshape(tf.range(count - np.product(shapes[-1]), count, dtype=tf.int32), shapes[-1])

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """
     
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                model.trainable_variables[i].assign(tf.reshape(param, shape))
            
            # Assign the lambda_value
            lambda_value.assign([params[-1][0]])

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
                params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_d = loss_domain()
                loss_s = loss_solution()
                loss = loss_d + loss_s

            # calculate gradients and convert to 1D tf.Tensor
            trainable_vars = model.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            f.iter.assign_add(1)
            if f.iter % 100 == 0:
                tf.print("Epoch:", f.iter, "\t Loss = ", loss, '\t Loss_domain = ', loss_d, '\t Loss_s = ', loss_s,
                         '\t lambda_value = ', lambda_value[0])

            # store loss value so we can retrieve later
            tf.py_function(f.history.append, inp=[[loss, loss_d, loss_s]], Tout=[])

            return loss, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []

        return f


    # =====================================
    # Function that checks the model type:
    # =====================================
    def check_model_type(model):

        if isinstance(model, tf.keras.models.Sequential):
            return "Sequential API"
        elif isinstance(model, tf.keras.Model):
            # Check if the model has custom layers (indicating model subclassing)
            if any(isinstance(layer, tf.keras.layers.Layer) for layer in model.layers):
                return "Subclassed Model"
            else:
                return "Functional API"
        else:
            return "Unknown"
