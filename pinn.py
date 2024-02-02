import numpy as np
import tensorflow as tf
import time
from utilities import Domain, PDE, PINN_function_factory


# ==============================================================================================================
# Physics-Informed Neural Network:
# ==============================================================================================================

class PINN:

    def __init__(self, model, optimizer, function_factory, loss, pde, X_domain, Y_domain, X_bc, Y_bc, X_ic, Y_ic):

      # Instantiate the metric trackers:
      self.loss_tracker= tf.keras.metrics.Mean(name='loss')
      self.bc_tracker = tf.keras.metrics.Mean(name="loss_bc")
      self.ic_tracker = tf.keras.metrics.Mean(name="loss_ic")
      self.domain_tracker = tf.keras.metrics.Mean(name="loss_domain")
      self.history = []

      # Pass the class arguments:
      self.model = model
      self.optimizer = optimizer
      self.function_factory = function_factory   # Function
      self.loss_domain, self.loss_bc, self.loss_ic = loss
      self.pde = pde   # Function
      self.X_domain = X_domain
      self.Y_domain = Y_domain
      self.X_bc = X_bc
      self.Y_bc = Y_bc
      self.X_ic = X_ic
      self.Y_ic = Y_ic

  
    # Build & compile the model:
    def prepare_model(self):
      self.model.build(input_shape=(None,2))
      self.model.compile()
      return self.model.summary()

    # Compute the PDE residuals loss: 
    def compute_loss_domain(self):

        u_domain_pred = self.pde(self.model, self.X_domain)
        loss_domain = self.loss_domain(self.Y_domain, u_domain_pred)
        return loss_domain

    # Compute the boundary conditions loss: 
    def compute_loss_bc(self):

        u_bc_pred = self.model(self.X_bc, training=True)
        loss_bc = self.loss_bc(self.Y_bc, u_bc_pred)
        return loss_bc

    # Compute the initial conditions loss: 
    def compute_loss_ic(self):

        u_ic_pred = self.model(self.X_ic, training=True)
        loss_ic = self.loss_ic(self.Y_ic, u_ic_pred)
        return loss_ic

  
    @tf.function
    def train_step(self):

        with tf.GradientTape(persistent=True) as tape:
          tape.watch(self.model.trainable_variables)
          
          # Compute the losses:
          loss_domain = self.compute_loss_domain()
          loss_bc = self.compute_loss_bc()
          loss_ic = self.compute_loss_ic()
          loss = loss_domain + loss_bc + loss_ic

        # Compute the gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.domain_tracker.update_state(loss_domain)
        self.bc_tracker.update_state(loss_bc)
        self.ic_tracker.update_state(loss_ic)
        self.loss_tracker.update_state(loss)

        return loss, loss_domain, loss_bc, loss_ic


    # Train with the defined optimizer:
    def standard_training(self, epochs):

      printing_step = 100
      for i in range(epochs + 1):

        loss, loss_domain, loss_bc, loss_ic = self.train_step()

        if i % printing_step == 0:
            print('Epoch: {}\t Loss = {}\t Loss_domain = {}\t Loss_bc = {}\t Loss_ic = {}'.format(i, self.loss_tracker.result(),
                                                                                                     self.domain_tracker.result(),
                                                                                                     self.bc_tracker.result(),
                                                                                                     self.ic_tracker.result()))
            self.history.append([i,
                                 self.loss_tracker.result(),
                                 self.domain_tracker.result(),
                                 self.bc_tracker.result(),
                                 self.ic_tracker.result()]

        # Reset the metrics
        for m in self.metrics:
                m.reset_states()

      # Return history:
      HIST = np.array(self.history).reshape((int(epochs/printing_step)+1, len(self.history[0])))
      return HIST

    
    # Train with the L-BFGS optimizer:
    def lbfgs_training(self):

        func = self.function_factory(self.model, self.compute_loss_domain, self.compute_loss_bc, self.compute_loss_ic)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            parallel_iterations=8, # Parallel iterations
            max_iterations=500 # Maximum number of iterations
        )
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        loss_lbfgs = np.array(func.history)
        #self.history = np.concatenate([np.array(self.history), loss_lbfgs], axis=0)
        print("L-BFGS best loss: {:2.4e}".format(loss_lbfgs[-1, 0]))

        epochs = np.arange(len(loss_lbfgs)).reshape((len(loss_lbfgs), 1))
        history = np.hstack([epochs, np.array(loss_lbfgs)])

        return history.reshape((len(loss_lbfgs), len(loss_lbfgs[0])+1))



    def hybrid_training(self, epochs):
        import time

        initial_time = time.time()
        # Training with adam
        print(" ====== Training with Adam ==========")
        history_adam = self.adam_training(epochs)

        # Training with lbfgs
        print(" ====== Training with L -BFGS ==========")
        history_lbfgs = self.lbfgs_training()

        final_time = time.time()
        print('\nComputation time: {} seconds\n'.format(round(final_time - initial_time)))

        total_history = np.vstack([history_adam, history_lbfgs])


        path = 'gdrive/MyDrive'
        #os.makedirs(path)
        self.model.save(f'{path}/saved_model')
        # Save loss history
        np.savetxt(f'{path}/total_history.txt', total_history)
        return total_history

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.domain_tracker, self.bc_tracker, self.ic_tracker]
