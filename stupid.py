import pygame

pygame.init()
import numpy as np
import copy

# np.random.seed(9983)

# Chargez le fichier audio
pygame.mixer.music.load("tracteur.wav")
pygame.mixer.music.play(-1)


def grid():
    import random
    #from playsound import playsound
    import datetime
    import time
    import cv2

    import curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.initscr()
    curses.start_color()

    def meme():  # Il y avait un bug quand un meme s'affichait, ça faisait que la map ne se mettait plus à jour, donc j'efface pr l'instant
        # image = cv2.imread("%d.jpg"%random.randint(1,5))
        # window_name = "image"
        # cv2.imshow(window_name, image)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) #On veut que l'image soit au top, devant les autres trucs
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        return "Commande executée"

    def update_p(letter, p, m, a):
        while True:
            if letter == "z" and p["y"] != 0 and m[p["y"] - 1][p["x"]] == 2:  # monter
                p["y"] = p["y"] - 1
                break
            if letter == "s" and p["y"] != (len(game_map) - 1) and m[p["y"] + 1][p["x"]] == 2:  # descendre
                p["y"] = p["y"] + 1
                break
            if letter == "q" and p["x"] != 0:
                p["x"] = p["x"] - 1
                break
            if letter == "d" and p["x"] != (len(game_map[0]) - 1):
                p["x"] = p["x"] + 1
                break
            stdscr.addstr(len(game_map) + 7, 0, "You can't go there, try to be a bit smarter ;)")
            stdscr.refresh()
            # Si le mouvement est impossible, on fait jouer un son particulier.
            letter = stdscr.getkey()
        return p

    def generate_random_map(size_map, proportion_ladder,
                            proportion_empty):  # proportion va etre un pourcentage, ,proportion_ladder
        # size map (n,m), n colonnes, m lignes.
        proportion_empty = 0
        L = []  # On crée la matrice de la map
        for i in range(0, size_map[1]):  # On crée d'abord la ligne
            L.append([])
            for j in range(0, size_map[0]):  # On ajoute le nombre de colonnes
                L[i].append(0)
        # On a désormais crée la matrice nulle de taille (n,m). On va désormais rajouter les échelles et les trous.
        # On parcourt alors de nouveau la liste. pour éviter de tout parcourir tout le temps, on voir sur chaque case si on rajoute quelque chose
        for i in range(0, size_map[1]):  # Ligne
            num_ladder = 0
            for j in range(0, size_map[0]):  # Colonne
                # On va commencer par les probabilités les plus petites.
                if np.random.randint(0, 1000) < int(proportion_empty * 1000):  # la proportion va etre une fréquence.
                    L[i][j] = 1  # trou
                    continue
                if j == size_map[1] - 1 and num_ladder == 0:
                    L[i][j] = 2  # échelle
                    num_ladder += 1
                if np.random.randint(0, 1000) < int(proportion_ladder * 1000):  # la proportion va etre une fréquence.
                    L[i][j] = 2  # échelle
                    num_ladder += 1

        return L

    # map=generate_random_map((int(input("Nombre de lignes de la carte")),int(input("Nombre de colonnes de la carte"))),float(input("Frequence echelle")),float(input("Frequence vide")))

    def display_map(m, d):  # m est la carte sous forme de matrice, #d est l'encodeur.
        # But : prend en paramètre une carte
        for i in range(0, len(m)):
            for j in range(0, len(m[i])):
                stdscr.addstr(i, j, d[m[i][j]])
            # print()

    def create_objects(nbobjects, m):
        obj_positions = {}
        for objet in range(nbobjects):
            y = random.randint(0, len(m) - 1)
            x = random.randint(0, len(m) - 1)
            couple = x, y
            obj_positions[couple] = 1
        return obj_positions

    def create_objects(nbobjects, m):
        obj_positions = {}
        for objet in range(nbobjects):
            y = random.randint(0, len(m) - 1)
            x = random.randint(0, len(m) - 1)
            while m[x][y] != 2:
                y = random.randint(0, len(m) - 1)
                x = random.randint(0, len(m) - 1)
            couple = x, y
            obj_positions[couple] = 1
        if len(obj_positions) != nbobjects:
            while len(obj_positions) != nbobjects:
                y = random.randint(0, len(m) - 1)
                x = random.randint(0, len(m) - 1)
                while m[x][y] != 2:
                    y = random.randint(0, len(m) - 1)
                    x = random.randint(0, len(m) - 1)
                couple = x, y
                if couple in obj_positions or m[x][y] == 1:
                    continue
                else:
                    obj_positions[couple] = 1
        return obj_positions

    def update_objects(p, objects):
        if (p["x"], p[
            "y"]) in objects:  # Si les coordonnées du joueur sont les mêmes que les coordonnées d'un objet, on supprime l'objet
            del objects[(p["x"], p["y"])]
            game_map[p["y"]][p["x"]] = 2
            p["score"] += 1
        return objects

    def create_perso(depart):
        d = {}
        d["x"] = depart[0]
        d["y"] = depart[1]
        d["repr"] = depart[2]
        d["score"] = int(0)
        return d

    def display_map_and_char(m, d, l,
                             objects):  # m correspond à la map, d correspond à l'encodage, l correspond à la liste de personnages
        for i in range(0, len(m)):  # i correspond aux y
            for j in range(0, len(m[i])):  # j correspond aux x
                b = False  # On suppose de base que c'est faux
                for k in l:  # je veux une petite instructio qui vérifie si un des deux personnages est à cet endroit. Si c'est pas le cas alors on met le else.
                    if k["x"] == j and k["y"] == i:  # si un personnage est à cet endroit
                        b = True
                        if k["repr"] == '\u2666':
                            stdscr.addstr(i, j, k["repr"], curses.color_pair(2))
                        else:
                            stdscr.addstr(i, j, k["repr"], curses.color_pair(3))
                        # print(k["repr"],end="")
                if not b:  # pas de personnage à cet endroit
                    if (j, i) in objects:
                        stdscr.addstr(i, j, chr(9774))

                        # print("o", end = "") # A chaque endroit de la map où il est sensé y avoir un objet, on affiche un objet représenté par "x"
                    else:
                        stdscr.addstr(i, j, d[m[i][j]])
                        # print(d[m[i][j]],end="")
            print()

    stdscr.addstr(0, 0, "Once upon a time, in a galaxy far, faaaaar away...")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(1, 0, "...A little guy : Constantin Philippe de la Martinière")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(2, 0,
                  "Constantin Philippe de la Martinière was very happy, he liked to watch Curling competitions (everyone does).")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(3, 0, "Sometimes he also liked to listen to Punk music !")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(4, 0, "Constantin Philippe de la Martinière loved everything about life.")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(5, 0,
                  "There is only one thing that make him angry : an animal from planet Earth called PEACE AND LOVE")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(6, 0, "One day, Constantin Philippe de la Martinière was attacked by a Peace and Love !")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(7, 0, "Now, he wants to eat the Peace and Love to get his revenge...")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(8, 0,
                  "Eat the Peace and Love! Go up and down using the ladders. This is a rapidity game, your goal is to be fast")
    stdscr.refresh()
    time.sleep(0.1)
    stdscr.addstr(10, 0, "Press C to continue, I to watch the AI play, and V to play against the AI")
    cont = stdscr.getkey()
    # Fin du storytelling, mise en place des données du jeu

    # def play_grid(id_game) :
    #    play=True
    #    while play == True :
    game_map = generate_random_map((10, 10), 0.4, 0)  # don't touch the number of lines and columns
    nbobjects = 1

    objects = create_objects(nbobjects, game_map)
    objects1 = copy.deepcopy(objects)

    for couple in objects.keys():
        x, y = couple
        game_map[x][y] = 2
        game_map[y][x] = 2

    if cont == "c":
        fin = 1
        # game_map=generate_random_map((10,10),0.4,0) #don't touch the number of lines and columns
        dico = {0: '_', 1: '\u25CB', 2: '#'}  # _ is the normal ground, 1 is the hole, 2 is the ladder
        # nbobjects = 1

        # objects=create_objects(nbobjects,game_map)
        p = create_perso((0, 0, '\u2666'))  #
        # q=create_perso((2,3,'\u2660'))#
        # l=[p,q] #On fait une liste avec tous les persos.
        l = [p]
        compteur_de_tours = 10
        liste_publicites = [
            "ADVERTISING: Are you also tired of having a stomach ache when you go into space aboard your Renault Twingo rocket? Do like me: get Staracadvomi anti-emetics! Staracadvomi, to support weightlessness without feeling dizzy ;) Order on our web page: https://renault-effects-secondaires/nicholas-latifi/espace/maux-de-ventre/staracadvomi.fr Medicine not to be used use in pregnant women and children under 12 years old. For more information, ask your doctor for advice.",
            "ADVERTISING: If you too are tired of hearing Renaud's song Très Debout, sign the petition: RENAUD MUST STOP HIS CAREER! Petition link: https://www.change.org/renault-c-etait-mieux-avant.fr",
            "ADVERTISING: Before, Nicolas ate cereal for breakfast. But... Since his sister Leonidastrovski introduced him to Mousline mash fries with their broccoli sauce, he can't live without them! More information on www://la-pataterie.fr"]

        # Début du jeu
        stdscr.erase()
        start_time = datetime.datetime.now()  # On chronomètre la durée de la partie
        fin = 0
        a = True
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        while a:
            stdscr.addstr(len(game_map) + 3, 0, "Player 1 score :")
            stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
            # stdscr.addstr(len(game_map)+4,0,"Player 2 score :")
            # stdscr.addstr(len(game_map)+4,25,str(l[1]["score"]))
            stdscr.addstr(len(game_map) + 6, 0, "Commands : ZQSD to move", curses.color_pair(1))

            display_map_and_char(game_map, dico, l, objects)
            stdscr.addstr(len(game_map) + 2, 0, "Player 1 turn")
            stdscr.refresh()
            d = stdscr.getkey()  # REPRENDRE ICI POUR SUPPRESSION DU JOUEUR 2

            l[0] = update_p(d, l[0], game_map, a)
            objects = update_objects(l[0], objects)
            display_map_and_char(game_map, dico, l, objects)

            stdscr.addstr(len(game_map) + 3, 0, "Player 1 score :")
            stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
            # stdscr.addstr(len(game_map)+4,0,"Player 2 score :")
            # stdscr.addstr(len(game_map)+4,25,str(l[1]["score"]))
            # stdscr.addstr(len(game_map)+2,0,"Player 2 turn")
            stdscr.addstr(len(game_map) + 6, 0, "Commands : ZQSD to move", curses.color_pair(1))

            stdscr.refresh()

            # d = stdscr.getkey() #move of player 2
            # if d == "r" :
            #    while d == "r" :
            #        map=generate_random_map((10,10),0.4,0.1)
            #        display_map_and_char(map,dico,l,objects)
            #        stdscr.refresh()
            #        d = stdscr.getkey()

            # l[1]=update_p(d,l[1],map,a)
            # objects = update_objects(l[1],objects)
            # display_map_and_char(map,dico,l,objects)

            # stdscr.refresh()

            if game_map[l[0]["y"]][l[0]["x"]] == 1:
                stdscr.addstr(len(game_map) + 5, 0, "You lost :( Give me 1000 bahts to play again.")
                stdscr.refresh()
                a = False
                break
            if compteur_de_tours % 10 == 0:
                numeropub = random.randint(0, 2)
                stdscr.addstr(len(game_map) + 8, 0, liste_publicites[
                    numeropub])  # On affiche au hasard une des 3 publicités tous les 10 tours
                stdscr.refresh()

            compteur_de_tours += 1

            if l[0]["score"] == nbobjects:  # Si tous les objets sont ramassés, le jeu est fini.
                fin = 1
                a = False
                break
            else:
                stdscr.erase()
                stdscr.refresh()
                continue

        # Fin du jeu. Crédits.
        stdscr.erase()
        # if fin==1:
        # stdscr.addstr(0,0,"All fragments have been collected! Well done.")
        # stdscr.refresh()
        # image = cv2.imread("reussi.jpg")
        # window_name = "image"
        # cv2.imshow(window_name, image)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) #On veut que l'image soit au top, devant les autres trucs
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        if fin == 0:
            stdscr.addstr(0, 0,
                          "You absolute pelican lmao. You have fallen into a trap! Yet it was in front of you and it was indicated! You had to use your critical thinking.")
            stdscr.refresh()
            # image = cv2.imread("echec.jpg")
            # window_name = "image"
            # cv2.imshow(window_name, image)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) #On veut que l'image soit au top, devant les autres trucs
            # cv2.waitKey(10000)
            # cv2.destroyAllWindows()
        else:
            stdscr.addstr(0, 0, "All fragments have been collected! Well done.")
            stdscr.refresh()

        end_time = datetime.datetime.now()

        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds()

        stdscr.addstr(2, 0, "Game time : ")
        stdscr.addstr(2, 16, str(execution_time))
        stdscr.refresh()
        time.sleep(5)

        # stdscr.addstr(0,0,"Press L to replay, and M to go to main menu")
        # stdscr.refresh()
        # o = stdscr.getkey() #move of player 1
        #    if o == "l" :
        #        play=True
        #    elif o == "m" :
        #        play=False
        #        id_game = 0 #id_game 0 is

        # On affiche la durée de la partie
        # time.sleep(1)
        # print("Crédits") #Les crédits s'affichent à  la fin du jeu.
        # time.sleep(1)
        # print("Graphismes : Edwin Desailly")
        # time.sleep(1)
        # print("Histoire : Mathis Do Cao")
        # time.sleep(1)
        # print("Développement : Edwin Desailly, Mathis Do Cao")
        # time.sleep(1)
        # print("Copie interdite. Tous droits réservés.")

    import random
    # distance between 2 moves is always 1
    list_path = []

    def detect_obstacle(pos, game_map):  # pos is a tuple, the position of player
        dont_go = []
        i, j = pos
        if j == 0:
            dont_go.append("left")
            if game_map[i][j + 1] == 1:
                dont_go.append("right")
        elif j == len(game_map[0]) - 1:  # at the right of the map
            dont_go.append("right")
            if game_map[i][j - 1] == 1:
                dont_go.append("left")
        else:
            # print(game_map[i][j-1])
            if game_map[i][j - 1] == 1:
                dont_go.append("left")
            if game_map[i][j + 1] == 1:
                dont_go.append("right")

        if i == 0:
            dont_go.append("up")
            if game_map[i + 1][j] != 2:
                dont_go.append("down")
        elif i == len(game_map) - 1:
            dont_go.append("down")
            if game_map[i - 1][j] != 2:  # si ce n'est pas une échelle au-dessus
                dont_go.append("up")
        else:
            if game_map[i - 1][j] != 2:  # si ce n'est pas une échelle au-dessus
                dont_go.append("up")
            if game_map[i + 1][j] != 2:
                dont_go.append("down")
        go = ["up", "down", "right", "left"]
        real_go = []
        for i in range(len(go)):
            if go[i] in dont_go:
                continue
            else:
                real_go.append(go[i])
        return real_go

    if cont == "i":

        def create_objects(nbobjects):  # function that generates objects at random positions
            obj_positions = {}
            for objet in range(nbobjects):
                y = random.randint(0, 10 - 1)
                x = random.randint(0, 10 - 1)
                couple = x, y
                obj_positions[couple] = 1
            if len(obj_positions) != nbobjects:
                while len(obj_positions) != nbobjects:
                    y = random.randint(0, 10 - 1)
                    x = random.randint(0, 10 - 1)
                    couple = x, y
                    obj_positions[couple] = 1
            return obj_positions

        # objects = create_objects(1)
        # objects1 = copy.deepcopy(objects)
        # game_map = generate_random_map((10,10),0.4,0)

        def initialize_q_table(state_space, action_space):
            Qtable = np.zeros((state_space, action_space))
            return Qtable

        Qtable = initialize_q_table(100, 4)  # 100 possible states (10*10) and 4 actions in each case
        # Training parameters
        n_training_episodes = 1000
        learning_rate = 0.7

        # Evaluation parameters
        n_eval_episodes = 1000

        # Environment parameters
        env_id = "cela ne sert pour rien"
        max_steps = 99
        gamma = 0.95
        eval_seed = []

        # Exploration parameters
        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.0005

        def detect_obstacle(pos, game_map):
            dont_go = []
            i, j = pos
            # print(i,j)
            # if i > 9 : print(i)
            # if i < -9 : print(i)
            if j <= 0:
                dont_go.append("left")
                if game_map[i][j + 1] == 1:
                    dont_go.append("right")
            elif j >= len(game_map[0]) - 1:  # at the right of the map
                dont_go.append("right")
                if game_map[i][j - 1] == 1:
                    dont_go.append("left")
            else:
                if game_map[i][j - 1] == 1:
                    dont_go.append("left")
                if game_map[i][j + 1] == 1:
                    dont_go.append("right")

            if i <= 0:
                dont_go.append("up")
                if game_map[i + 1][j] != 2:
                    dont_go.append("down")
            elif i == len(game_map) - 1:
                dont_go.append("down")
                if game_map[i - 1][j] != 2:  # si ce n'est pas une échelle au-dessus
                    dont_go.append("up")
            else:
                if game_map[i - 1][j] != 2:  # si ce n'est pas une échelle au-dessus
                    dont_go.append("up")
                if game_map[i + 1][j] != 2:
                    dont_go.append("down")
            go = ["up", "down", "right", "left"]
            real_go = []
            for i in range(len(go)):
                if go[i] in dont_go:
                    continue
                else:
                    real_go.append(go[i])
            return real_go

        def random_action(game_map, state):  # 0 monter, 1 descendre, 2 gauche, 3 droite
            y = state // 10
            x = state % 10
            cango = detect_obstacle([y, x], game_map)
            action_space = []
            for i in range(len(cango)):
                if cango[i] == "up":
                    action_space.append(0)
                elif cango[i] == "down":
                    action_space.append(1)
                elif cango[i] == "left":
                    action_space.append(2)
                elif cango[i] == "right":
                    action_space.append(3)
            action = random.choice(action_space)
            return action

        def epsilon_greedy_policy(Qtable, state, epsilon):
            random_int = random.uniform(0, 1)
            if random_int > epsilon:
                y = state // 10
                x = state % 10
                cango = detect_obstacle([y, x],
                                        game_map)  # detect_obstacle will give a list of possible moves in the position of the player

                action_space = []
                for i in range(len(cango)):
                    if cango[i] == "up":
                        action_space.append(0)
                    elif cango[i] == "down":
                        action_space.append(1)
                    elif cango[i] == "left":
                        action_space.append(2)
                    elif cango[i] == "right":
                        action_space.append(3)

                action = np.argmax(Qtable[state])
                if action not in action_space:
                    action = random_action(game_map, state)
            else:
                # action = env.action_space.sample() #ENV ICI : fonction qui choisit au hasard une action dans les actions possibles (dans cango)
                action = random_action(game_map, state)
            return action

        def greedy_policy(Qtable, state):
            action = np.argmax(Qtable[state])
            return action

        # créer une fonction qui retourne : nvl environnement, le reward, True or False si un objet est récolté
        def step_fun(action, objects, state):
            # we assume that the action is possible
            count = 0
            while True:
                count += 1
                if action == 0:  # go up
                    state -= 10
                    # p["y"]=p["y"]-1

                    break
                if action == 1:  # go down
                    state += 10
                    # p["y"]=p["y"]+1
                    break
                if action == 2:  # go left
                    state -= 1
                    # p["x"]=p["x"]-1
                    break
                if action == 3:  # go right
                    state += 1
                    # p["x"]=p["x"]+1
                    break
            object_collected = False
            y = state // 10  # peut-être inverser x et y ?
            x = state % 10
            if (x,
                y) in objects:  # Si les coordonnées du joueur sont les mêmes que les coordonnées d'un objet, on supprime l'objet
                reward = 100 / count
                # game_map[p["y"]][p["x"]]=0
                # p["score"] += 1
                object_collected = True
            else:
                reward = -count  # devrait peut-être pas être négatif
            return state, reward, object_collected

        def reset_situation():
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            # p["x"]=x
            # p["y"]=y
            state = y * 10 + x
            # passer de state à coordonnées : y = state // 10, x = state%10
            return state

        def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable):
            for episode in range(n_training_episodes):
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
                # Reset the environment
                state = reset_situation()
                step = 0
                done = False
                # repeat
                for step in range(max_steps):
                    action = epsilon_greedy_policy(Qtable, state, epsilon)
                    y = state // 10
                    x = state % 10
                    cango = detect_obstacle([y, x], game_map)
                    action_space = []
                    for i in range(len(cango)):
                        if cango[i] == "up":
                            action_space.append(0)
                        elif cango[i] == "down":
                            action_space.append(1)
                        elif cango[i] == "left":
                            action_space.append(2)
                        elif cango[i] == "right":
                            action_space.append(3)
                    if action not in action_space:
                        action = random_action(game_map,
                                               state)  # ENV ICI : créer une fonction qui retourne : nvl environnement, le reward, True or False si un objet est récolté
                    new_state, reward, done = step_fun(action, objects,
                                                       state)  # DEFINIR LES VARIABLES GAMEMAP OBJECTS P
                    Qtable[state][action] = Qtable[state][action] + learning_rate * (
                                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
                    if done:
                        break
                    state = new_state
            return Qtable

        Qtable_display = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable)

        def create_perso(depart):
            d = {}
            d["x"] = depart[0]
            d["y"] = depart[1]
            d["repr"] = depart[2]
            d["score"] = int(0)
            return d

        def display(Qtable, objects, game_map):
            player = create_perso([0, 0, "X"])
            state = 0
            old_state = -178
            list_states = [0]
            count = 0
            while len(objects) > 0:
                action = int(np.argmax(Qtable[state][:]))
                y = state // 10
                x = state % 10

                cango = detect_obstacle([y, x], game_map)

                action_space = []
                for i in range(len(cango)):
                    if cango[i] == "up":
                        action_space.append(0)
                    elif cango[i] == "down":
                        action_space.append(1)
                    elif cango[i] == "left":
                        action_space.append(2)
                    elif cango[i] == "right":
                        action_space.append(3)
                if action not in action_space:
                    action = random_action(game_map, state)
                if count >= 300:
                    k = random.random()
                    if k > 0.3:
                        action = random_action(game_map, state)

                new_state, reward, done = step_fun(action, objects, state)
                # while new_state>99 or new_state<0:
                # action = np.argmax(Qtable[state][:])
                #    action = random_action(game_map,state)
                # print("infinite loop detector")
                # print(action)
                #    new_state, reward, done = step_fun(action,objects,state)

                list_states.append(new_state)
                y = new_state // 10
                x = new_state % 10
                if (x, y) in objects:
                    del objects[(x, y)]
                # print(new_state)
                # print(x,y,objects)

                old_state = state
                state = new_state
                count += 1

            return list_states

        list_states = display(Qtable_display, objects, game_map)

        # print(game_map)
        def state_to_pos(list_states):
            list_pos = []
            for state in list_states:
                state_str = str(state)
                if state < 10:
                    y = 0
                    x = int(state_str[0])
                else:

                    x = int(state_str[1])
                    y = int(state_str[0])
                list_pos.append([x, y])
            return list_pos

        list_pos = state_to_pos(list_states)

        def display_game(list_pos, objects1):
            stdscr.erase()

            map1 = game_map

            dico = {0: '_', 1: '\u25CB', 2: '#'}  # _ is the normal ground, 1 is the hole, 2 is the ladder
            nbobjects = 1
            p1 = create_perso(
                (0, 0, '\u2666'))  # les joueurs ne spawnent pas au bon endroit lors de l'affichage de la partie
            p = p1.copy()
            objects = copy.deepcopy(objects1)
            l = [p]  # On fait une liste avec tous les persos.
            start_time = datetime.datetime.now()  # On chronomètre la durée de la partie
            fin = 1
            a = True
            curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)

            stdscr.erase()

            for pos in list_pos:
                stdscr.addstr(len(game_map) + 3, 0, "AI's score :")
                stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
                objects = update_objects(l[0], objects)
                display_map_and_char(game_map, dico, l, objects)
                stdscr.refresh()

                # Player 1
                time.sleep(0.3)
                l[0]["y"] = pos[1]
                l[0]["x"] = pos[0]

                objects = update_objects(l[0], objects)
                display_map_and_char(game_map, dico, l, objects)

                stdscr.addstr(len(game_map) + 3, 0, "AI's score :")
                stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
                stdscr.refresh()
            stdscr.erase()
            end_time = datetime.datetime.now()

            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds()

            stdscr.addstr(2, 0, "Game time : ")
            stdscr.addstr(2, 16, str(execution_time))
            stdscr.refresh()
            time.sleep(4)

        display_game(list_pos, objects1)

    # play against the AI
    if cont == "v":
        # create the list of moves of the AI
        def create_objects(nbobjects):  # function that generates objects at random positions
            obj_positions = {}
            for objet in range(nbobjects):
                y = random.randint(0, 10 - 1)
                x = random.randint(0, 10 - 1)

                couple = x, y
                obj_positions[couple] = 1
            if len(obj_positions) != nbobjects:
                while len(obj_positions) != nbobjects:
                    y = random.randint(0, 10 - 1)
                    x = random.randint(0, 10 - 1)
                    couple = x, y
                    obj_positions[couple] = 1
            return obj_positions

        # objects = create_objects(1)
        # objects1 = copy.deepcopy(objects)
        # game_map = generate_random_map((10,10),0.4,0)

        def initialize_q_table(state_space, action_space):
            Qtable = np.zeros((state_space, action_space))
            return Qtable

        Qtable = initialize_q_table(100, 4)  # 100 possible states (10*10) and 4 actions in each case
        # Training parameters
        n_training_episodes = 1000
        learning_rate = 0.7

        # Evaluation parameters
        n_eval_episodes = 1000

        # Environment parameters
        env_id = "cela ne sert pour rien"
        max_steps = 99
        gamma = 0.95
        eval_seed = []

        # Exploration parameters
        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.0005

        def detect_obstacle(pos, game_map):
            dont_go = []
            i, j = pos
            # print(i,j)
            # if i > 9 : print(i)
            # if i < -9 : print(i)
            if j <= 0:
                dont_go.append("left")
                if game_map[i][j + 1] == 1:
                    dont_go.append("right")
            elif j >= len(game_map[0]) - 1:  # at the right of the map
                dont_go.append("right")
                if game_map[i][j - 1] == 1:
                    dont_go.append("left")
            else:
                if game_map[i][j - 1] == 1:
                    dont_go.append("left")
                if game_map[i][j + 1] == 1:
                    dont_go.append("right")

            if i <= 0:
                dont_go.append("up")
                if game_map[i + 1][j] != 2:
                    dont_go.append("down")
            elif i == len(game_map) - 1:
                dont_go.append("down")
                if game_map[i - 1][j] != 2:  # si ce n'est pas une échelle au-dessus
                    dont_go.append("up")
            else:
                if game_map[i - 1][j] != 2:  # si ce n'est pas une échelle au-dessus
                    dont_go.append("up")
                if game_map[i + 1][j] != 2:
                    dont_go.append("down")
            go = ["up", "down", "right", "left"]
            real_go = []
            for i in range(len(go)):
                if go[i] in dont_go:
                    continue
                else:
                    real_go.append(go[i])
            return real_go

        def random_action(game_map, state):  # 0 monter, 1 descendre, 2 gauche, 3 droite
            y = state // 10
            x = state % 10
            cango = detect_obstacle([y, x], game_map)
            action_space = []
            for i in range(len(cango)):
                if cango[i] == "up":
                    action_space.append(0)
                elif cango[i] == "down":
                    action_space.append(1)
                elif cango[i] == "left":
                    action_space.append(2)
                elif cango[i] == "right":
                    action_space.append(3)
            action = random.choice(action_space)
            return action

        def epsilon_greedy_policy(Qtable, state, epsilon):
            random_int = random.uniform(0, 1)
            if random_int > epsilon:
                y = state // 10
                x = state % 10
                cango = detect_obstacle([y, x], game_map)

                action_space = []
                for i in range(len(cango)):
                    if cango[i] == "up":
                        action_space.append(0)
                    elif cango[i] == "down":
                        action_space.append(1)
                    elif cango[i] == "left":
                        action_space.append(2)
                    elif cango[i] == "right":
                        action_space.append(3)

                action = np.argmax(Qtable[state])
                if action not in action_space:
                    action = random_action(game_map, state)
            else:
                # action = env.action_space.sample() #ENV ICI : fonction qui choisit au hasard une action dans les actions possibles (dans cango)
                action = random_action(game_map, state)
            return action

        def greedy_policy(Qtable, state):
            action = np.argmax(Qtable[state])
            return action

        # créer une fonction qui retourne : nvl environnement, le reward, True or False si un objet est récolté
        def step_fun(action, objects, state):
            # we assume that the action is possible
            count = 0
            while True:
                count += 1
                if action == 0:  # go up
                    state -= 10
                    # p["y"]=p["y"]-1

                    break
                if action == 1:  # go down
                    state += 10
                    # p["y"]=p["y"]+1
                    break
                if action == 2:  # go left
                    state -= 1
                    # p["x"]=p["x"]-1
                    break
                if action == 3:  # go right
                    state += 1
                    # p["x"]=p["x"]+1
                    break
            object_collected = False
            y = state // 10  # peut-être inverser x et y ?
            x = state % 10
            if (x,
                y) in objects:  # Si les coordonnées du joueur sont les mêmes que les coordonnées d'un objet, on supprime l'objet
                reward = 100 / count
                # game_map[p["y"]][p["x"]]=0
                # p["score"] += 1
                object_collected = True
            else:
                reward = -count
            return state, reward, object_collected

        def reset_situation():
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            # p["x"]=x
            # p["y"]=y
            state = y * 10 + x
            # passer de state à coordonnées : y = state // 10, x = state%10
            return state

        def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable):

            for episode in range(n_training_episodes):

                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
                # Reset the environment
                # state = env.reset() #ENV ICI --> créer une fonction qui place aléatoirement sur la carte
                state = reset_situation()
                step = 0
                done = False

                # repeat
                for step in range(max_steps):

                    action = epsilon_greedy_policy(Qtable, state, epsilon)
                    y = state // 10
                    x = state % 10

                    cango = detect_obstacle([y, x], game_map)

                    action_space = []
                    for i in range(len(cango)):
                        if cango[i] == "up":
                            action_space.append(0)
                        elif cango[i] == "down":
                            action_space.append(1)
                        elif cango[i] == "left":
                            action_space.append(2)
                        elif cango[i] == "right":
                            action_space.append(3)
                    if action not in action_space:
                        action = random_action(game_map, state)

                    # new_state, reward, done, info = env.step(action) #ENV ICI : créer une fonction qui retourne : nvl environnement, le reward, True or False si un objet est récolté
                    new_state, reward, done = step_fun(action, objects,
                                                       state)  # DEFINIR LES VARIABLES GAMEMAP OBJECTS P
                    # while new_state>99 or new_state<0:
                    #    action = random_action(game_map,state) #problem : des fois on doit sûrement faire des moves illégaux (monter alors que pas le droit), d'où les chiffres négatifs
                    #    new_state, reward, done = step_fun(action,objects,state)

                    Qtable[state][action] = Qtable[state][action] + learning_rate * (
                                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

                    # If done, finish the episode
                    if done:
                        break

                    # Our state is the new state
                    state = new_state
            return Qtable

        Qtable_display = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable)

        def create_perso(depart):
            d = {}
            d["x"] = depart[0]
            d["y"] = depart[1]
            d["repr"] = depart[2]
            d["score"] = int(0)
            return d

        def display(Qtable, objects, game_map):
            player = create_perso([0, 0, "X"])
            state = 0
            old_state = -178
            list_states = [0]
            count = 0
            while len(objects) > 0:
                action = int(np.argmax(Qtable[state][:]))
                y = state // 10
                x = state % 10

                cango = detect_obstacle([y, x], game_map)

                action_space = []
                for i in range(len(cango)):
                    if cango[i] == "up":
                        action_space.append(0)
                    elif cango[i] == "down":
                        action_space.append(1)
                    elif cango[i] == "left":
                        action_space.append(2)
                    elif cango[i] == "right":
                        action_space.append(3)
                if action not in action_space:
                    action = random_action(game_map, state)
                if count >= 300:
                    k = random.random()
                    if k > 0.3:
                        action = random_action(game_map, state)

                new_state, reward, done = step_fun(action, objects, state)
                # while new_state>99 or new_state<0:
                # action = np.argmax(Qtable[state][:])
                #    action = random_action(game_map,state)
                # print("infinite loop detector")
                # print(action)
                #    new_state, reward, done = step_fun(action,objects,state)

                list_states.append(new_state)
                y = new_state // 10
                x = new_state % 10
                if (x, y) in objects:
                    del objects[(x, y)]
                # print(new_state)
                # print(x,y,objects)

                old_state = state
                state = new_state
                count += 1

            return list_states

        list_states = display(Qtable_display, objects, game_map)

        # print(game_map)
        def state_to_pos(list_states):
            list_pos = []
            for state in list_states:
                state_str = str(state)
                if state < 10:
                    y = 0
                    x = int(state_str[0])
                else:

                    x = int(state_str[1])
                    y = int(state_str[0])
                list_pos.append([x, y])
            return list_pos

        list_pos = state_to_pos(list_states)

        def display_game(list_pos, objects1):
            stdscr.erase()

            map1 = game_map

            dico = {0: '_', 1: '\u25CB', 2: '#'}  # _ is the normal ground, 1 is the hole, 2 is the ladder
            nbobjects = 1
            p1 = create_perso(
                (0, 0, '\u2660'))  # les joueurs ne spawnent pas au bon endroit lors de l'affichage de la partie
            p = p1.copy()
            objects = copy.deepcopy(objects1)
            # On fait une liste avec tous les persos.
            start_time = datetime.datetime.now()  # On chronomètre la durée de la partie
            fin = 1
            a = True

            ai_win = 1
            # objects_player = copy.deepcopy(objects)
            p_player = create_perso((0, 0, '\u2666'))
            l = [p, p_player]

            curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)

            stdscr.erase()
            start = 0
            if len(list_pos) > 1:
                start = 1

            for pos in list_pos[start:]:
                stdscr.addstr(len(game_map) + 3, 0, "AI's score :")
                stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
                stdscr.addstr(len(game_map) + 6, 0, "Player's score :")
                stdscr.addstr(len(game_map) + 6, 25, str(l[1]["score"]))
                stdscr.addstr(len(game_map) + 9, 0, "Commands : ZQSD to move", curses.color_pair(1))
                stdscr.addstr(len(game_map) + 12, 0, "AI's turn")
                objects = update_objects(l[0], objects)
                display_map_and_char(game_map, dico, l, objects)
                stdscr.refresh()

                # Player 1
                time.sleep(0.2)
                l[0]["y"] = pos[1]
                l[0]["x"] = pos[0]

                objects = update_objects(l[0], objects)
                display_map_and_char(game_map, dico, l, objects)

                stdscr.addstr(len(game_map) + 3, 0, "AI's score :")
                stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
                stdscr.addstr(len(game_map) + 6, 0, "Player's score :")
                stdscr.addstr(len(game_map) + 6, 25, str(l[1]["score"]))
                stdscr.addstr(len(game_map) + 9, 0, "Commands : ZQSD to move", curses.color_pair(1))
                stdscr.addstr(len(game_map) + 12, 0, "Player's turn")
                stdscr.refresh()

                d = stdscr.getkey()  # REPRENDRE ICI POUR SUPPRESSION DU JOUEUR 2
                l[1] = update_p(d, l[1], game_map, a)
                objects = update_objects(l[1], objects)
                display_map_and_char(game_map, dico, l, objects)

                stdscr.addstr(len(game_map) + 3, 0, "AI's score :")
                stdscr.addstr(len(game_map) + 3, 25, str(l[0]["score"]))
                stdscr.addstr(len(game_map) + 6, 0, "Player's score :")
                stdscr.addstr(len(game_map) + 6, 25, str(l[1]["score"]))
                stdscr.addstr(len(game_map) + 9, 0, "Commands : ZQSD to move", curses.color_pair(1))
                stdscr.addstr(len(game_map) + 12, 0, "AI's turn")
                stdscr.refresh()

                if l[1]["score"] == 1:
                    ai_win = 0
                    break

            if ai_win == 1:
                stdscr.erase()
                end_time = datetime.datetime.now()

                time_diff = (end_time - start_time)
                execution_time = time_diff.total_seconds()
                stdscr.addstr(5, 0,
                              "The AI won ! Machines are going to invade the world, prepare yourself for apocalypse.")
                stdscr.addstr(2, 0, "Game time : ")
                stdscr.addstr(2, 16, str(execution_time))
                stdscr.refresh()
                time.sleep(5)
            if ai_win == 0:
                stdscr.erase()
                end_time = datetime.datetime.now()

                time_diff = (end_time - start_time)
                execution_time = time_diff.total_seconds()
                stdscr.addstr(5, 0, "You won ! Congratulations, you have just won your Computer Science degree.")
                stdscr.addstr(2, 0, "Game time : ")
                stdscr.addstr(2, 16, str(execution_time))
                stdscr.refresh()
                time.sleep(5)

        display_game(list_pos, objects1)


from math import *


def ball():
    import pygame

    def start():
        pygame.init()

        screen = pygame.display.set_mode((1300, 700))
        window = Window(screen)
        window.run()
        pygame.quit()

    class Window:

        def __init__(self, screen):

            self.screen = screen
            self.background_color = (0, 0, 0)
            self.clock = pygame.time.Clock()
            self.mouse_x = 100
            self.mouse_y = 100

            self.player = Player()
            self.ground_level = 500
            self.time = 0

        def handling_events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:  # Check if the left arrow key is pressed
                        self.a -= 0.2
                        print(self.a)
                    if event.key == pygame.K_RIGHT:  # Check if the right arrow key is pressed
                        self.a += 0.2
                        print(self.a)
                    if event.key == pygame.K_DOWN:  # Check if the up arrow key is pressed
                        self.b -= 0.2
                        print(self.b)
                    if event.key == pygame.K_UP:  # Check if the down arrow key is pressed
                        self.b += 0.2
                        print(self.b)
            for event in pygame.mouse.get_pressed():
                if event:
                    self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

                    self.player.x0, self.player.y0 = self.mouse_x, self.mouse_y
                    self.player.x, self.player.y = self.mouse_x, self.mouse_y
                    self.player.vx0 = self.a * 200  # vitesse horizontale
                    self.player.vy0 = -self.b * 200  # vitesse verticale
                    self.player.start_time = self.time

        def update(self):

            self.player.update(self.ground_level, self.time)

            self.score = ceil(self.player.score_update(int(self.score)))

        def display(self):
            self.screen.fill(self.background_color)
            pygame.draw.circle(self.screen, (200, 200, 200), (self.mouse_x, self.mouse_y), 10)
            self.player.display(self.screen)
            pygame.draw.line(self.screen, (255, 255, 255), (0, self.ground_level), (1300, self.ground_level), 10)

            pygame.font.init()  ## INITIALIZE FONT
            myfont = pygame.font.SysFont('monospace', 30)

            a_display = myfont.render("Horizontal : " + str(round(self.a, 2)), False, (120, 187, 0))
            self.screen.blit(a_display, (5, 5))  ## Blit rendu Font
            b_display = myfont.render("Vertical : " + str(round(self.b, 2)), False, (120, 187, 0))
            self.screen.blit(b_display, (5, 40))  ## Blit rendu Font
            score_display = myfont.render("Score : " + str(self.score), False, (120, 187, 0))
            self.screen.blit(score_display, (5, 80))  ## Blit rendu Font
            pygame.draw.rect(self.screen, (120, 187, 0),
                             (1290, 300, 10, 200))  # Position : 1250,300 ; Dimensions : 50,200
            pygame.display.flip()

        def run(self):
            self.running = True
            self.a = 3
            self.b = 3
            self.score = 0
            while self.running:
                self.handling_events()
                self.update()
                self.display()
                self.clock.tick(60)
                self.time += 1 / 60

    class Player:

        def __init__(self):

            self.start_time = 0

            self.x = 0
            self.y = 0

            self.x0 = 0
            self.y0 = 0

            self.vx0 = 0
            self.vy0 = 0

            self.ax0 = 0
            self.ay0 = 9.18 * 300

        def update(self, ground_level, time):

            if self.y < ground_level:
                t = time - self.start_time
                self.x = self.vx0 * t + self.x0
                self.y = (1 / 2) * self.ay0 * t ** 2 + self.vy0 * t + self.y0  # ax^2 + bx + c

        def score_update(self, score):
            if 300 < self.y < 500 and 1290 < self.x < 1300:
                score += 0.1
            return score

            # Pour détecter quand ça touche le but : if self.x (condition avec une coordonnée) and self.y (condition) :
            # ajouter 1 au score etc

        def display(self, screen):
            pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), 40)

    start()


def home():
    import random
    # from playsound import playsound
    import datetime
    import time
    import cv2

    import curses
    homescr = curses.initscr()
    curses.noecho()
    curses.initscr()
    curses.start_color()

    on = True
    while on:
        homescr.erase()
        homescr.addstr(0, 0, "To play Déplace-toi allègrement sur la grille de bon matin, press A")
        homescr.addstr(2, 0, "To play Lance des boules avec beaucoup d'amusement, press B")
        homescr.addstr(4, 0, "To stop the program :'(, press Z")
        homescr.refresh()
        d = homescr.getkey()  # move of player 2
        homescr.erase()
        if d == "a":
            grid()
            homescr.erase()
        elif d == "b":
            ball()
            homescr.erase()
        elif d == "z":
            on = False


home()









