import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

# Call the main function to get the animation object
print("Generating animation...")
multi_agent_ani = main_multi_agent()

# Save the animation to a file
# Using 'ffmpeg' writer, which needs to be installed (e.g., sudo apt-get install ffmpeg)
# For Google Colab, ffmpeg is usually pre-installed.
print("Saving animation to multi_agent_animation.mp4...")
writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
multi_agent_ani.save('multi_agent_animation.mp4', writer=writer)
print("Animation saved!")

# Display the animation in the notebook
print("Displaying animation...")
display(HTML(multi_agent_ani.to_jshtml()))
