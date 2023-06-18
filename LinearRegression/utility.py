import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib import animation
from IPython import display
import numpy as np


def add_axis_for_bias(X_i):
    
    X_i = X_i.copy()
    if len(X_i.shape) == 1:
        X_i = X_i.reshape(-1,1)

    if False in (X_i[...,0] == 1):        
        return np.hstack(tup=(np.ones(shape=(X_i.shape[0],1)) , X_i))
    else:
        return X_i
    

def predict(theta, X):
    format_X = add_axis_for_bias(X)
        
    if format_X.shape[1] == theta.shape[0]:
        y_pred = format_X @ theta # (m,1) = (m,n) * (n,1)
        return y_pred
    elif format_X.shape[1] == theta.shape[1]:
        y_pred = format_X @ theta.T # (m,1) = (m,n) * (n,1)
        return y_pred
    else:
        raise ValueError("Shape is not proper.")
        
def regression_animation(X, y, cost_history, theta_history, iterations, interval=20):
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2,3,1)
    ax4 = fig.add_subplot(2,3,2, projection="3d")
    ax3 = fig.add_subplot(2,3,3)
    ax2 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,(5,6))

    min_cost, max_cost = cost_history.min(), cost_history.max()
    min_theta, max_theta = theta_history.min(), theta_history.max()

    ax1.set(xlim=(0, len(cost_history)), ylim=(min_cost, max_cost),
            xlabel='iterations', ylabel='loss', title='Loss over iterations')
    ax2.set(title='Regression line')
    ax3.set(xlim=(min_theta, max_theta), ylim=(min_cost, max_cost),
            xlabel='theta', ylabel='loss', title='Loss vs theta 2D')
    ax4.set(xlim=(min_theta, max_theta), ylim=(min_theta, max_theta), zlim=(min_cost,max_cost), 
            xlabel='theta0', ylabel='theta1', zlabel='loss', title='Loss vs theta 3D')
                  
    line1, = ax1.plot([],[])
    ax2.scatter(X,y)
    line2, = ax2.plot([],[],'k')
    line31, = ax3.plot([],[],'.-')
    line32, = ax3.plot([],[],'.-')
    line4, = ax4.plot([],[],[], '.-', color='black')
    
    ax5.plot(range(len(y)),y, color='k')
    line5, = ax5.plot([],[], '.-', alpha=0.8)
    fig.tight_layout()

    def draw_frame(n):
        y_pred = predict(theta_history[n],X)
        
        ax1.set(title=f"Loss over iterations : {round(cost_history[n], 3)}")
        line1.set_data(range(n),cost_history[:n])
        
        line2.set_data(X, y_pred)
        
        line31.set_data(theta_history[:n,0],cost_history[:n])
        line32.set_data(theta_history[:n,1],cost_history[:n])
        
        line4.set_data_3d(theta_history[:n,0], theta_history[:n,1], cost_history[:n])
        
        line5.set_data(range(len(y)), y_pred)
        return (line1,line2,line31,line32,line4,line5, )

    anim = animation.FuncAnimation(fig, draw_frame, frames=iterations, interval=interval, blit=True)
    return display.HTML(anim.to_html5_video())


def regression_plot(X, y, cost_history, theta_history, iterations):

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2,3,1)
    ax4 = fig.add_subplot(2,3,2, projection="3d")
    ax3 = fig.add_subplot(2,3,3)
    ax2 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,(5,6))

    min_cost, max_cost = cost_history.min(), cost_history.max()
    min_theta, max_theta = theta_history.min(), theta_history.max()

    ax1.set(xlim=(0, len(cost_history)), ylim=(min_cost, max_cost),
            xlabel='iterations', ylabel='loss', title='Loss over iterations')
    ax2.set(title='Regression line')
    ax3.set(xlim=(min_theta, max_theta), ylim=(min_cost, max_cost),
            xlabel='theta', ylabel='loss', title='Loss vs theta 2D')
    ax4.set(xlim=(min_theta, max_theta), ylim=(min_theta, max_theta), zlim=(min_cost,max_cost), 
            xlabel='theta0', ylabel='theta1', zlabel='loss', title='Loss vs theta 3D')

    ax1.plot(cost_history)
    ax2.scatter(X, y)

    y_pred = predict(theta_history[-1], X)
    ax2.plot(X, y_pred, 'k')

    ax3.plot(theta_history[:, 0], cost_history, '.-')
    ax3.plot(theta_history[:, 1], cost_history, '.-')
    ax4.plot(theta_history[:, 0], theta_history[:, 1], cost_history, '.-', color='black')

    ax5.plot(y, color='k')
    ax5.plot(y_pred, '.-', alpha=0.8)
    
    fig.tight_layout()
    return fig