"""
Prospectors n Pirates - Quick Launcher

This script provides an easy menu to run different examples and tests.
"""

import os
import sys


def print_banner():
    print("\n" + "=" * 70)
    print("  PROSPECTORS N PIRATES - Game Simulation")
    print("  Deep Reinforcement Learning Environment")
    print("=" * 70)


def print_menu():
    print("\nWhat would you like to do?\n")
    print("  [1] Run Environment Tests")
    print("  [2] Basic Demo (Random & Rule-based Agents)")
    print("  [3] Advanced Customization Examples")
    print("  [4] Train DQN Agent (Custom Implementation)")
    print("  [5] Train with Stable Baselines3 (PPO)")
    print("  [6] Train with Stable Baselines3 (DQN)")
    print("  [7] Compare RL Algorithms")
    print("  [8] Quick Test (25k steps)")
    print("  [9] View Documentation")
    print("  [0] Exit")
    print()


def run_command(cmd):
    """Run a command and wait for it to complete"""
    print(f"\nRunning: {cmd}\n")
    os.system(cmd)
    print("\n" + "=" * 70)
    input("Press Enter to continue...")


def main():
    while True:
        print_banner()
        print_menu()

        choice = input("Enter your choice [0-9]: ").strip()

        if choice == '1':
            run_command("python test_env.py")

        elif choice == '2':
            run_command("python example_basic.py")

        elif choice == '3':
            run_command("python example_advanced.py")

        elif choice == '4':
            print("\n" + "=" * 70)
            print("Training DQN Agent")
            print("=" * 70)
            print("This will train a custom DQN implementation for 500 episodes.")
            print("Estimated time: 20-30 minutes on CPU")
            confirm = input("\nContinue? [y/N]: ").strip().lower()
            if confirm == 'y':
                run_command("python example_dqn.py")

        elif choice == '5':
            print("\n" + "=" * 70)
            print("Training PPO Agent with Stable Baselines3")
            print("=" * 70)
            print("This will train a PPO agent for 100,000 timesteps.")
            print("Estimated time: 45 minutes on CPU")
            confirm = input("\nContinue? [y/N]: ").strip().lower()
            if confirm == 'y':
                run_command("python example_sb3.py --algorithm PPO --timesteps 100000")

        elif choice == '6':
            print("\n" + "=" * 70)
            print("Training DQN Agent with Stable Baselines3")
            print("=" * 70)
            print("This will train a DQN agent for 100,000 timesteps.")
            print("Estimated time: 30 minutes on CPU")
            confirm = input("\nContinue? [y/N]: ").strip().lower()
            if confirm == 'y':
                run_command("python example_sb3.py --algorithm DQN --timesteps 100000")

        elif choice == '7':
            print("\n" + "=" * 70)
            print("Comparing RL Algorithms")
            print("=" * 70)
            print("This will train PPO, DQN, and A2C for 50,000 timesteps each.")
            print("Estimated time: 1-2 hours")
            confirm = input("\nContinue? [y/N]: ").strip().lower()
            if confirm == 'y':
                run_command("python example_sb3.py --algorithm compare --timesteps 50000")

        elif choice == '8':
            print("\n" + "=" * 70)
            print("Quick Test Training")
            print("=" * 70)
            print("This will quickly train an A2C agent for 25,000 timesteps.")
            print("Estimated time: 5-10 minutes on CPU")
            confirm = input("\nContinue? [y/N]: ").strip().lower()
            if confirm == 'y':
                run_command("python example_sb3.py --algorithm A2C --timesteps 25000")

        elif choice == '9':
            print("\n" + "=" * 70)
            print("Documentation")
            print("=" * 70)
            print("\nAvailable documentation files:")
            print("  1. README.md - Complete documentation")
            print("  2. GETTING_STARTED.md - Beginner tutorial")
            print("  3. PROJECT_SUMMARY.md - Project overview")
            print("\nWhich would you like to view?")
            doc_choice = input("Enter 1, 2, or 3 (or press Enter to skip): ").strip()

            if doc_choice == '1':
                if os.path.exists('README.md'):
                    run_command("type README.md" if os.name == 'nt' else "cat README.md")
            elif doc_choice == '2':
                if os.path.exists('GETTING_STARTED.md'):
                    run_command("type GETTING_STARTED.md" if os.name == 'nt' else "cat GETTING_STARTED.md")
            elif doc_choice == '3':
                if os.path.exists('PROJECT_SUMMARY.md'):
                    run_command("type PROJECT_SUMMARY.md" if os.name == 'nt' else "cat PROJECT_SUMMARY.md")

        elif choice == '0':
            print("\n" + "=" * 70)
            print("Thanks for using Prospectors n Pirates!")
            print("=" * 70)
            print()
            sys.exit(0)

        else:
            print("\nInvalid choice. Please enter a number between 0 and 9.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
