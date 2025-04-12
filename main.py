import os
from simulation.core import CDPRSimulation
from simulation.viewer import CDPRViewer
from simulation.controller import CDPRController
from utils.cable_calculations import compute_cable_lengths

def main():
    # Initialize simulation
    model_path = os.path.join('models', 'four_tendons.xml')
    simulation = CDPRSimulation(model_path)
    viewer = CDPRViewer(simulation.model)
    controller = CDPRController(simulation.model, simulation.data)
    
    # Main loop
    while not viewer.should_close() and simulation.data.time < simulation.simend:
        # Update controller
        controller.update()
        
        # Step simulation
        simulation.step()
        
        # Update viewer
        viewer.update(simulation.model, simulation.data)
    
    viewer.close()

if __name__ == '__main__':
    main()