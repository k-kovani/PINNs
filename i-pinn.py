import numpy as np
import time
import tensorflow as tf
import tensorflow_probability as tfp
from utilities import PINN_function_factory, check_model_type, save_model
from PDEs import pde
#from geometries import geom


# ==============================================================================================================
# Inverse Physics-Informed Neural Network:
# ==============================================================================================================


class IPINN:

    def __init__(self, model, optimizer, IPINN_function_factory, losses, pde, X_solution, Y_solution, X_domain, Y_domain, lambda_value):

      # Initialize the metric trackers:
      self.loss_tracker= tf.keras.metrics.Mean(name='loss')
      self.solution_tracker = tf.keras.metrics.Mean(name="loss_solution")
      self.domain_tracker = tf.keras.metrics.Mean(name="loss_domain")
      self.history = []

      # Pass the class arguments:
      self.model = model
      self.optimizer = optimizer
      self.function_factory = IPINN_function_factory  # Function
      self.loss_solution, self.loss_domain = losses
      self.pde = pde  # Function
      self.X_solution = X_solution
      self.Y_solution = Y_solution
      self.X_domain = X_domain
      self.Y_domain = Y_domain
      self.N_inputs = X_domain.shape[1]

      # Register the lambda_value as a model trainable parameter:
      self.lambda_value = tf.Variable([lambda_value], dtype=tf.float32, trainable=True)
  

 
    # Build and compile the model:
    def prepare_model(self):
      self.model.build(input_shape=(None,self.N_inputs))
      self.model.compile()
      return self.model.summary()


    # Compute the PDE residual loss:
    def compute_loss_domain(self):

        u_domain_pred = self.pde(self.model, self.X_domain, self.lambda_value)
        loss_domain = self.loss_domain(self.Y_domain, u_domain_pred)
        return loss_domain


    # Compute the solution prediction loss:
    def compute_loss_solution(self):

        u_solution_pred = self.model(self.X_solution, training=True)
        loss_bc = self.loss_solution(self.Y_solution, u_solution_pred)
        return loss_bc


    @tf.function
    def train_step(self):

        with tf.GradientTape(persistent=True) as tape:
          tape.watch(self.model.trainable_variables)
          
          # Coompute the losses:
          loss_domain = self.compute_loss_domain()
          loss_solution = self.compute_loss_solution()
          loss = loss_domain + loss_solution

        # Compute the gradients:
        trainable_vars = self.model.trainable_variables
        trainable_vars.append(self.lambda_value)      # IMPORTANT! Add the lambda_value in trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.domain_tracker.update_state(loss_domain)
        self.solution_tracker.update_state(loss_solution)
        self.loss_tracker.update_state(loss)

        return loss, loss_domain, loss_solution

  
    # Train with the defined optimizer:
    def standard_training(self, epochs):

      # Keep track of time:
      initial_time = time.time()
      
      printing_step = 100
      for i in range(epochs + 1):

        loss, loss_domain, loss_solution = self.train_step()

        if i % printing_step == 0:
            print('Epoch: {}\t Loss = {}\t Loss_domain = {}\t Loss_solution = {}\t mu = {}\t lambda_value = {}'.format(i,
                                                                                                                       self.loss_tracker.result(),
                                                                                                                       self.domain_tracker.result(),
                                                                                                                       self.solution_tracker.result(),
                                                                                                                       viscocity_coefficient,
                                                                                                                       self.lambda_value.numpy()[0]))
            self.history.append([i,
                                 self.loss_tracker.result(),
                                 self.domain_tracker.result(),
                                 self.solution_tracker.result()])
        # Reset the metrics
        for m in self.metrics:
                m.reset_states()
      
      # Print the training time:
      final_time = time.time()
      print('\nTotal Computation time: {} seconds\n'.format(round(final_time - initial_time)))
      
      # Return the history:
      HIST = np.array(self.history).reshape((int(epochs/printing_step)+1, len(self.history[0])))
      return HIST


    # Train with the L-BFGS optimizer:
    def lbfgs_training(self):
        
        # Check for model compatibility:
        if check_model_type(self.model) == "Subclassed Model":
            print("Error! The L-BFGS optimizer in TensorFlow 2.0 is not designed to work with Subclassed models. Please use the Sequential or Functional API.")
        else:
            if self.function_factory is None:
                print("Error: you didn't provide the function factory")
            else:
                
                # Call the function_factory:
                func = self.function_factory(self.model, self.compute_loss_domain, self.compute_loss_solution, self.lambda_value)
        
                # convert initial model parameters to a 1D tf.Tensor
                init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)

                # Keep record of time:
                initial_time = time.time()
                
                # train the model with L-BFGS solver
                results = tfp.optimizer.lbfgs_minimize(
                    value_and_gradients_function=func,
                    initial_position=init_params,
                    parallel_iterations=10,     # Parallel iterations
                    max_iterations=500          # Maximum number of iterations
                )
          
                # Print the training time:
                final_time = time.time()
                print('\nTotal Computation time: {} seconds\n'.format(round(final_time - initial_time)))
                
                # Extract the final lambda_value from the optimized parameters
                final_lambda_value = results.position[-1]
    
                # Assign the final lambda_value to the model
                self.lambda_value.assign([final_lambda_value])
    
                # after training, the final optimized parameters are still in results.position
                # Manually put them back to the model
                func.assign_new_model_parameters(results.position)

                # Print the best loss:
                loss_lbfgs = np.array(func.history)
                print("L-BFGS best loss: {:2.4e}".format(loss_lbfgs[-1, 0]))

                # Return the history:
                epochs = np.arange(len(loss_lbfgs)).reshape((len(loss_lbfgs), 1))
                history = np.hstack([epochs, np.array(loss_lbfgs)])
                HIST = history.reshape((len(loss_lbfgs), len(loss_lbfgs[0])+1))
                return HIST



    def hybrid_training(self, epochs):
        
        # Keep track of time:
        initial_time = time.time()
        
        # Training with defined optimizer:
        print(" ====== Training with optimizer ==========")
        history_optimizer = self.standard_training(epochs)

        # Refining with L-BFGS
        print(" ====== Refining with L-BFGS ==========")
        history_lbfgs = self.lbfgs_training()
        
        # Print the training time:
        final_time = time.time()
        print('\nTotal Computation time: {} seconds\n'.format(round(final_time - initial_time)))

        # Save the model:
        path = './saved_model'
        save_model(path)

        # Return the history:
        total_history = [history_optimizer, history_lbfgs]
        return total_history

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.domain_tracker, self.solution_tracker]




# ========================================================
#  Example: Compute the viscocity in 1D Burger's Equation 
# ========================================================

# Create the training data:
# ---------------------------
X_domain, Y_domain = geom.
X_solution = np.load('./X_sol_data.npy')
Y_solution = np.load('./Y_sol_data.npy')

# Define the pde:
# -----------------
pde = pde.1D_Burgers()
lambda_value = 2.0    # Solution's (true) viscosity = 0.2

# Define the model:
# ---------------------------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(20, "tanh"))
model.add(tf.keras.layers.Dense(20, "tanh"))
model.add(tf.keras.layers.Dense(30, "tanh"))
model.add(tf.keras.layers.Dense(30, "tanh"))
model.add(tf.keras.layers.Dense(20, "tanh"))
model.add(tf.keras.layers.Dense(1, "tanh"))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
losses = [tf.keras.losses.MeanSquaredError(), # loss domain
        tf.keras.losses.MeanSquaredError()] # loss solution

# Instantiate the IPINN:
pinn = IPINN(model, optimizer, IPINN_function_factory, losses, pde, X_solution, Y_solution, X_domain, Y_domain, lambda_value)
pinn.prepare_model()

# Train the PINN:
history = pinn.hybrid_training(1000)
