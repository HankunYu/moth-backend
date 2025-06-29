import numpy as np
import cv2
import time
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import os

class GeneticMaskBlender:
    def __init__(self, image1_path, image2_path, mask_resolution=(50,50), 
                 population_size=20, generations=50, mutation_rate=0.1):
        # Load images
        self.image1 = cv2.imread(image1_path)
        # self.image1 = cv2.resize(self.image1, (512, 512)) 
        self.image2 = cv2.imread(image2_path)
        # self.image2 = cv2.resize(self.image2, (512, 512)) 
        
        # Validate images
        if self.image1 is None or self.image2 is None:
            raise ValueError("Could not load one or both images")
        if self.image1.shape != self.image2.shape:
            raise ValueError("Images must be the same dimensions")
            
        # Parameters
        self.mask_res = mask_resolution
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.img_height, self.img_width = self.image1.shape[:2]
        
        # Tracking
        self.best_fitness_history = []
        self.processing_time = 0
        
    def initialize_population(self):
        """Initialize population of random masks"""
        return [np.random.rand(*self.mask_res) for _ in range(self.pop_size)]
    
    def resize_mask(self, mask):
        """Resize mask to match image dimensions"""
        return cv2.resize(mask, (self.img_width, self.img_height), 
                         interpolation=cv2.INTER_LINEAR)
    
    def blend_images(self, mask):
        """Blend images using the mask"""
        resized_mask = self.resize_mask(mask)
        resized_mask_3d = np.stack([resized_mask]*3, axis=-1)  # For color images
        return (self.image1 * resized_mask_3d + self.image2 * (1 - resized_mask_3d)).astype(np.uint8)
    
    def fitness_function(self, blended_img):
        """Evaluate blend quality using multiple metrics"""
        # Convert images to grayscale for SSIM
        gray_blended = cv2.cvtColor(blended_img, cv2.COLOR_BGR2GRAY)
        gray_img1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM with both images
        ssim1, _ = ssim(gray_img1, gray_blended, full=True)
        ssim2, _ = ssim(gray_img2, gray_blended, full=True)
        
        # Edge preservation (using Laplacian variance)
        laplacian = cv2.Laplacian(gray_blended, cv2.CV_64F)
        edge_score = np.var(laplacian)
        
        # Combine metrics (weights can be adjusted)
        return 0.4*ssim1 + 0.4*ssim2 + 0.2*(edge_score/1000)
    
    def tournament_selection(self, population, fitness, tournament_size=3):
        """Select parents using tournament selection"""
        selected = []
        for _ in range(len(population)):
            contestants = np.random.choice(len(population), tournament_size)
            winner = contestants[np.argmax([fitness[i] for i in contestants])]
            selected.append(population[winner].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        """Two-point crossover for masks"""
        if np.random.rand() < 0.8:  # 80% crossover probability
            h, w = parent1.shape
            # Select two random crossover points
            cx1, cx2 = sorted([np.random.randint(0, w), np.random.randint(0, w)])
            cy1, cy2 = sorted([np.random.randint(0, h), np.random.randint(0, h)])
            
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            # Swap the rectangular region between the points
            child1[cy1:cy2, cx1:cx2] = parent2[cy1:cy2, cx1:cx2]
            child2[cy1:cy2, cx1:cx2] = parent1[cy1:cy2, cx1:cx2]
            
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Apply random mutations to the mask"""
        mutated = individual.copy()
        mutation_mask = np.random.rand(*self.mask_res) < self.mutation_rate
        noise = np.random.normal(0, 0.2, size=self.mask_res)
        mutated[mutation_mask] = np.clip(mutated[mutation_mask] + noise[mutation_mask], 0, 1)
        return mutated
    
    def run(self):
        """Execute the genetic algorithm"""
        start_time = time.time()
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness = []
            for ind in population:
                blended = self.blend_images(ind)
                fitness.append(self.fitness_function(blended))
            
            self.best_fitness_history.append(np.max(fitness))
            
            # Selection
            parents = self.tournament_selection(population, fitness)
            
            # Create next generation
            next_population = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    next_population.extend([self.mutate(child1), self.mutate(child2)])
                else:
                    next_population.append(self.mutate(parents[i]))
            
            # Elitism - keep best individual
            best_idx = np.argmax(fitness)
            next_population[0] = population[best_idx].copy()
            population = next_population
            
            # Progress reporting
            if generation % 5 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness_history[-1]:.4f}")
        
        # Get best solution
        final_fitness = [self.fitness_function(self.blend_images(ind)) for ind in population]
        best_idx = np.argmax(final_fitness)
        best_mask = population[best_idx]
        best_blend = self.blend_images(best_mask)
        
        self.processing_time = time.time() - start_time
        
        print(f"\nOptimization complete in {self.processing_time:.2f} seconds")
        print(f"Best fitness score: {final_fitness[best_idx]:.4f}")
        
        return best_mask, best_blend
    
    def visualize_results(self, best_mask, best_blend):
        """Display the results"""
        plt.figure(figsize=(15, 5))
        
        # Original images
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB))
        plt.title("Image 1")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB))
        plt.title("Image 2")
        plt.axis('off')
        
        # Evolved mask
        plt.subplot(1, 4, 3)
        plt.imshow(self.resize_mask(best_mask), cmap='gray', vmin=0, vmax=1)
        plt.title("Evolved Blending Mask")
        plt.axis('off')
        
        # Blended result
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(best_blend, cv2.COLOR_BGR2RGB))
        plt.title("Blended Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Fitness progression
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_history)
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
        in_dir = './inputs/'
        in_files = os.listdir(in_dir)
        out_dir = './out_textures/'
        os.makedirs(out_dir, exist_ok=True)
        num_files = len(in_files)
    # try:
        # Initialize blender
        for i in range(num_files):
            for j in range(i + 1, num_files):
                # if not i == j:
                    name1 = in_files[i]
                    name2 = in_files[j]
                    print('img1 %s, img2 %s'%(name1, name2))
                    blender = GeneticMaskBlender(in_dir+name1, in_dir+name2, 
                                            mask_resolution=(20,20),  # Lower resolution = faster evolution
                                            population_size=15,
                                            generations=30,
                                            mutation_rate=0.15)
                    
                    # Run the genetic algorithm
                    best_mask, best_blend = blender.run()
                    
                    # Visualize results
                    # blender.visualize_results(best_mask, best_blend)
                    
                    # Save results
                    # cv2.imwrite(out_dir+name1[:-4]+name2, (blender.resize_mask(best_mask)*255).astype(np.uint8))
                    out_img = cv2.resize(best_blend, (512, 512))
                    cv2.imwrite(out_dir+name1[:-4]+'-'+name2, out_img)
                
    # except Exception as e:
    #     print(f"Error: {str(e)}")