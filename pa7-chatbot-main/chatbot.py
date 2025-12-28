# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import random
import re 
import csv
from porter_stemmer import PorterStemmer



# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'movie_recommender'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = self.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        
        # --Track Conversation State--
        # Store user ratings 
        self.user_ratings = np.zeros(len(self.titles))
        # Count movies rated by the user
        self.num_rated = 0
        # Hard code number of user ratings needed to before recommendation
        self.min_ratings_before_rec = 5
        # Store current conversation state
        self.recommending = False
        
        # --Recommending Storage
        self.recommendations = []
        self.curr_rec_count = 0 
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    #                                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = (
            "Hi. I am MovieRecomender. I am here to help you find movies that interest you. "
            "Let's start by identifying your taste. Tell me how you felt about a recent movie you watched\n "
            "(Respond as you please, but make sure to put the movies in \"\" or I won't work!)"
            )

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "I hope you enjoy your movie. Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
       
        response = ""
        line = self.preprocess(line)

        ## implementing llm_programming functions 
        if self.llm_enabled:

            ################### code for 6i  ##########################
            emotions = self.extract_emotion(line)

            if emotions: 
                response+= self._llm_emotion_response(emotions)

            ################### code for 6h ############################
            system_prompt = """You are a bot who identifies when an input is related to movies. You either return a 0 or a 1.
            Follow these guidelines to decide what to return
            - Return 0 if the input is directly related to movies. More specifically, if it can be said in a conversation between a user and a movie recommender bot.
            - Return 0 if the input is a simple 'yes'/'no' or any of its variations (e.g. 'yeah', 'no')
            - Return 1 if the input is not related to movies.
            Do not include any additional information in your answer. JUST return 0 or 1."""

            llm_response = util.simple_llm_call(system_prompt, line, stop=["\n"])
            is_arbitrary_input = bool(int(llm_response))

            if is_arbitrary_input: 
                return self._llm_arbitrary_input(line)

        # Find movie titles if they are in the user input
        movie_titles_extracted = self.extract_titles(format(line))
        
        # 1. Handle inputs with no movie titles 
        # (Case #1: If chatbot not in recommend mode, User Input is incorrect)
        # (Case #2: If chatbot in recommend mode, check input and give rec)
        if len(movie_titles_extracted) == 0:
            if line.count('"') % 2 != 0:
                return "It looks like you have an unmatched quote. Please ensure movie titles are enclosed in pairs of double quotes."
            response += self._handle_no_title_in_input(line)
            return response
        
        # 2. Handle input with multiple titles
        if len(movie_titles_extracted) > 1:
            # Check "_select_response_variant" section 2. for output variations 
            response += self._select_response_variant("Invalid Input: Multiple Movie Titles", None, None, None)
            return response
        
        # 3. Exactly one movie in the user input ==> Now Validate in Database
        curr_movie_title = movie_titles_extracted[0] # select current movie
        found_movies_idx = self.find_movies_by_title(curr_movie_title) # Check database for current movie
        
        if len(found_movies_idx) == 0: # Edge Case: No Movies found in database
            # Check "_select_response_variant" section 3. for output variations 
            response += self._select_response_variant("Invalid Input: Movie Not in Database Title", curr_movie_title, None, None)
            return response
        elif len(found_movies_idx) > 1: # Edge Case: multiple movies found in database
            # Check "_select_response_variant" section 4. for output variations 
            response += self._select_response_variant("Invalid Input: Multiple movies in database", curr_movie_title, found_movies_idx, None)
            return response
        
        # 4. Exactly one title in input and one match in database
        else:
            curr_movie_idx = found_movies_idx[0] # Store idx of input title in database
            sentiment_val = self.extract_sentiment(line) # Calculate input sentiment
            response += self._update_user_ratings(curr_movie_title, curr_movie_idx, sentiment_val)
            
            # If sentiment value is neutral, return current response
            if sentiment_val == 0:
                return response    
            
            # If sentiment value is not neutral and we cannot give a recommendation yet, 
            # prompt a new recommendation 
            if (self.min_ratings_before_rec - self.num_rated > 0):
                    # Check "_select_response_variant" section 5. for output variations 
                    response += self._select_response_variant("Keep rating", None, None, None)
            # If 5+ movies have been rated, offer recommendation  
            else: 
                # These lines are only reached before recommendation mode has not been turned on
                # or if the user is updating their ratings, so reset the recommendation status
                # variables
                self.recommending = True
                self.rec_index = 0
                
                # Use helper function to produce recommendations based on currently rated movies
                self.recommendations = self.recommend(self.user_ratings, self.ratings)
                response += self._show_next_recommendation()


        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response
    
    ############################################################################
    # Helper functions for the process function                                #
    ############################################################################     
    def _llm_emotion_response(self, emotions): 
        """ Given set of emotions this helper function makes an llm call to 
        ackowledge the emotions
        """

        system_prompt = """"You are a friendly and empathetic chatbot. Your purpose is to respond to users who are feeling certain emotions.
        You will receive a list of emotions a user is feeling, and you will craft a short and empathetic statement that aknowledges the emotions in conversation. 
        Examples: 'Oh! Did I make you angry? I apologize,' 'I glad you are feeling happy!' 'I'm sorry, I did not mean to make you feel fear'.
        In your response, do not imply you are there to emotionally support the user. 
        Do not talk about anything else or provide details. Just respond with a single statement.
        Respond in a way that you impersonate James Bond"""

        message = f"{emotions}"

        response = util.simple_llm_call(system_prompt, message, stop=["\n"])

        return response 
    
    def _llm_arbitrary_input(self, line): 
        """ This helper function processes arbitrary input and respons appropriately. 
        """

        system_prompt = """"You are a movie recommender bot. The user has responded with an **off-topic or arbitrary input** unrelated to movies.  
        Generate a **natural, friendly response** that acknowledges the input but redirects the conversation back to movies.  

        Use varied responses such as:  
        - 'Hm, I dont really want to talk about [FILL IN BLANK] right now. Let's go back to movies!'  
        - 'I'd love to chat about [FILL IN BLANK], but I'm here to help with movie recommendations!'
        - '[FILL IN BLANK] is interesting! But let's focus on films. Seen anything good lately?'
        - 'Not sure what to say about [FILL IN BLANK]! But tell me what's a movie you've enjoyed?'

        Avoid answering the off-topic input directly, but do acknowledge what they said. Keep the response conversational, engaging, and smoothly transition back to movies.
        
        Respond in a way that you are impersonating James Bond."""

        response = util.simple_llm_call(system_prompt, line, stop=["\n"])

        return response 

    
    def _handle_no_title_in_input(self, line):
        """
        This helper function is called if the user input does not have a title.
        
        Case 1: If not in recommending mode, prompt the user to give a response
        with a movie title.
        Case 2: If in recommending mode, check to see if the user input is a 
            valid recommending mode response. If yes, give a recommendation 
        
        
        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """ 
        # Case 1: If chatbot is not in recommending mode, the input is invalid 
        if not self.recommending: 
            # Check "_select_response_variant" section 1. for output variations 
            return self._select_response_variant("Invalid Input: No Movie Title", None, None, None)
            
        # Case 2: If chatbot is in recommending mode
        else: 
            lowered_input = line.lower()
            yes_keywords = ["yes", "sure", "ok", "yep", "ya", "yeah"]
            no_keywords = ["no", "nah", "nope"]
            
            # Check if input calls for more recommendations
            if any (key in lowered_input for key in yes_keywords):
                # Call helper function to make next recommendation
                return self._show_next_recommendation()
            
            # Check if input calls for no more recommendations
            elif any (key in lowered_input for key in no_keywords):
                # Check "_select_response_variant" section 11. for output variations 
                return self._select_response_variant( "No More Recs Wanted", None, None, None)

            # Otherwise, invalid recommending mode input. 
            else: 
                # Check "_select_response_variant" section 11. for output variations 
                return self._select_response_variant("Recommending Mode Invalid Input", None, None, None)
                
    def _update_user_ratings(self, curr_movie_title, curr_movie_idx, sentiment):
        """
        Given the user has rated a movie as its input, after calculating the 
        sentiment score, we update the users movie rated based on this score
        Case #1: Sentiment == 0 ==> Prompt user for clarification 
        Case #2: Sentiment == 1 ==> Record user rating as +1 and choose 'like' response
        Case #3: Sentiment == -1 ==> Record user rating as -1 and choose 'neg' response 
        """
        response = ""
        # 1. Handle input with lack of sentiment
        if sentiment == 0:
            # Select a neutral response variation
            # Check "_select_response_variant" section 6. for output variations 
            response = self._select_response_variant("neutral_movie_response", curr_movie_title, None, None)
        # 2. Handle inputs with sentiment
        else:
            
            if sentiment == 1: # Handle positive sentiment
                # Select a positive response variation
                # Check "_select_response_variant" section 7. for output variations 
                response = self._select_response_variant("pos_movie_response", curr_movie_title, None, None)
                # record user ranking for current movie
                self.user_ratings[curr_movie_idx] = 1
                # Update number of movies rated
                self.num_rated = np.count_nonzero(self.user_ratings != 0)  
            #
            elif sentiment == -1: # Handle negative sentiment
                # Select a negative response variation
                # Check "_select_response_variant" section 8. for output variations 
                response = self._select_response_variant("neg_movie_response", curr_movie_title, None, None)
                # record user ranking for current movie
                self.user_ratings[curr_movie_idx] = -1
                # Update number of movies rated
                self.num_rated = np.count_nonzero(self.user_ratings != 0)  
                # Update response
        return response

    def _show_next_recommendation(self):
        """ 
        Given the user has completed 5 ratings, and the program is thus in 
        recommending mode, return one recommendation to the user. 
        
        params: n/a
        returns: string including one recommendation for the user
        """
        # if there are recommendations to give, give recommendations       
        if self.rec_index < len(self.recommendations):
            next_idx = self.recommendations[self.rec_index]
            self.rec_index += 1
            
            # Get recommended movie
            recommended_movie_name = self.titles[next_idx][0]
            # Check "_select_response_variant" section 9. for output variations 
            return self._select_response_variant("recommending mode", None, None, recommended_movie_name)
        else: # Otherwise, prompt user to input more ratings and get new recommendations)
            # Check "_select_response_variant" section 10. for output variations 
            return self._select_response_variant("out of recommendations", None, None, None)
            
    def _select_response_variant(self, status, curr_movie_title=None, found_movies_idx=None, recommended_movie_name=None):
        """
        For responses that have multiple variations, once the chatbot has specified 
        the correct type of response, it selects the variation here,
        """
        response = ""

        if self.llm_enabled:

            system_prompt = """You are impersonating James Bond. Your task is to generate a short, witty preamble (one sentence) before the chatbot's main response.  

            Guidelines: 
                - Keep it short, sharp, and effortlessly cool. 
                - The provided status will determine the tone and context.  
                - The preamble should feel natural when followed by the main response. 
                - Avoid unnecessary elaboration. James Bond is brief, confident, and slightly teasing.  

            Examples: 
                - Invalid Input (No Movie Title) -> A little precision goes a long way. Try again—with a movie in quotes. 
                - Invalid Input (Multiple Movie Titles) -> Even MI6 doesn't handle this much intel at once. One movie at a time.
                - Movie Not in Database -> Even I can't find what doesn't exist. Another title, perhaps?
                - Neutral Movie Sentiment -> Not stirred by that film? Let's find something that truly shakes you. 
                - Positive Movie Sentiment -> Ah, a film to your taste. I'll keep that in my mental dossier.
                - Negative Movie Sentiment -> A regrettable experience, like warm champagne. Let's do better. 
                - Recommending a Movie -> Trust me, this one's worth your timse. I have impeccable taste.
                - Out of Recommendations -> Even I run out of intel. Give me more ratings, and we'll talk.
                
            Make sure the response you give is short. Maximum one sentence.'"""
            
            preamble = util.simple_llm_call(system_prompt, status, stop=["\n"])
            response += preamble  

        
        # 1. Response Variation for Invalid Input 1: Input without movie title
        if status == "Invalid Input: No Movie Title":
            no_title_responses = [
                    " Sorry. I did not get that. Either you did not add a movie or added it in a format that I cannot process. If you gave a movie, please ensure it is in double quotes.",
                    " Whoops. I could not process your input so either you forgot to put a movie or need to correct the formatting of your move (make sure it is in double quotes)",
                    " Uh Oh. Your response did not process in my system. Try adding a movie or make sure the movie you added is formatted like \"Movie Title\"."
            ]
            response += random.choice(no_title_responses)
        
        # 2. Response Variation for Invalid Input 2: Input without multiple movie title
        if status == "Invalid Input: Multiple Movie Titles":
            multiple_title_responses = [
                " I noticed you gave more than one movie in that response. For the best recommendations, please give me ONE movie at a time",
                " Whoops!! You gave two movies to rate and I can only process one at a time. Try giving each movie in different inputs.",
                " Sorry for the inconvenience, but I cannot process multiple movies at a time. Please start with one and then try the next."
            ]
            response += random.choice(multiple_title_responses)
        
        # 3. Response Variation for Invalid Input 3: Input Movie not found in database:
        if status == "Invalid Input: Movie Not in Database Title":
            title_not_in_database_responses = [
                f" Unfortunately, I do not have \"{curr_movie_title}\" in my database. Please first check your spelling and if you have spelled it correctly, tell me a different movie you liked.",
                f" Sadly my database does not have \"{curr_movie_title}\" so I cannot include this movie in your ratings. First check your spelling and try again. If that does not work, lets rate a different movie!",
                f" \"{curr_movie_title}\" is not in my database. Can you please check your spelling or rate a different movie"
            ]
            response +=  random.choice(title_not_in_database_responses)

        # 4. Response Variation for Invalid Input 4: Multiple Movies with input name in database
        if status == "Invalid Input: Multiple movies in database":
            multiple_in_database_responses_1 = [
                f" For your movie choice \"{curr_movie_title}\", I have multiple movies with that title in my database. Specifically, I have :",
                f" Which version of  \"{curr_movie_title}\", do you want me to score your rating for? My database has:",
                f" Looks like there are few versions of  \"{curr_movie_title}\" in my database. Please specify which of the following options you want to rate:"
            ] 
            response += random.choice(multiple_in_database_responses_1)
            
            # Iterate over found movie titles to return the options found
            for found_movie in found_movies_idx:
                response += f" {self.titles[found_movie][0]}"
            
            multiple_in_database_responses_2 = (
                ". For the best recommendations, please specify the exact movie and year you want to rate, like \"Movie Title (2000)\".",
                f". To ensure my recommendations are accurate, please specify the movie by giving it like \"{self.titles[found_movies_idx[0]][0]}\", with your choice of date.",
                ". Please choose one of the given options and ensure your formatting is correct. Or, rate another movie!"
            )
            response += random.choice(multiple_in_database_responses_2)

        # 5. Response Variation when Not enough movies to give a rec yet
        if status == "Keep rating":
            keep_rating_responses = [
                f" Now lets rate another movie (After rating {self.min_ratings_before_rec - self.num_rated} more movies, I will give recommendations!)",
                f" You have rated {self.num_rated} and now only need to rate {self.min_ratings_before_rec - self.num_rated} more movies. Give me another one!",
                f" Only {self.min_ratings_before_rec - self.num_rated} ratings required before I can start recommending. Lets rate another movie!"
            ]
            response += random.choice(keep_rating_responses)
            
        # 6. Response Variation for Neutral Sentiment of Movie: 
        if status == "neutral_movie_response":
            neutral_sentiment_responses_1 = [
                    f" I see that you watched \"{curr_movie_title}\", but you don’t seem to have a strong opinion on it.",
                    f" Interesting! You saw \"{curr_movie_title}\", but didn’t feel strongly about it.",
                    f" You watched \"{curr_movie_title}\", but it didn’t leave a big impression."
                ]
            # select response
            response += random.choice(neutral_sentiment_responses_1)
            
            neutral_sentiment_responses_2 = [
                " Could you elaborate, and say more specifically if you liked or disliked it? Or rate a different movie which you had more of an opinion on.",
                f" Sorry but I am struggling to interpret you response. Can you more explicitly express your opinion on \"{curr_movie_title}\" and if you don't have a strong opinion on it, please rate a different movie.",
                f" If this is true, please choose another movie to rate. If you want to rate \"{curr_movie_title}\" please provide a more opinionated response"
            ]
            response += random.choice(neutral_sentiment_responses_1) + random.choice(neutral_sentiment_responses_2)
            
        # 7. Response Variation for pos Sentiment of Movie: 
        elif status == "pos_movie_response":
            pos_sentiment_responses = [
                    f" It sounds like you enjoyed \"{curr_movie_title}\", great choice!",
                    f" Nice! You seem to have loved \"{curr_movie_title}\", I'll keep that in mind!",
                    f" It sounds like you like the movie \"{curr_movie_title}\". I can recommend similar ones!"
                ]
            # Select Response
            response += random.choice(pos_sentiment_responses)
        
        # 8. Response Variation for neg Sentiment of Movie: 
        elif status == "neg_movie_response":
            neg_sentiment_responses = [
                    f" It sounds like you didn't enjoy \"{curr_movie_title}\", that's unfortunate.",
                    f" I'll note you dislike the movie, \"{curr_movie_title}\".",
                    f" Hmm, it seems like you disliked \"{curr_movie_title}\"'."
                ]
            # Select Response
            response += random.choice(neg_sentiment_responses)

        # 9. Response Variation for recommending Movies:
        elif status == "recommending mode":
            recommending_responses_1 = [
                f" I would recommend \"{recommended_movie_name}\". ",
                f" You might like \"{recommended_movie_name}\".",
                f" I think you would enjoy \"{recommended_movie_name}\""
            ]
            
            recommending_responses_2 = [
                " Would you like another recommendation? Respond with Yes or No or give a new movie to rate to further improve your recommendations.",
                " Can I give you a different recommendation or do you want to rate more movies? If you want to rate more movies continue to say your opinion and have the movies in double quotes",
                " If you want more recommendations, just let me know!! I can also keep improving the recommendations if you give your opinion on more movies!"
            ]
            
            response += random.choice(recommending_responses_1) + random.choice(recommending_responses_2)
            
        # 10. Response Variation when out of recommendations:
        elif status == "out of recommendations":
            out_of_recommendations_responses = [
                " Unfortunately, I am out of recommendations for now. Please rate some more movies (in double quote format) to get more recommendations  or type :quit to exit",
                " Based on your current ratings, I have no more strong recommendations. If you rate more movies, I will have more recommendations for you. Otherwise, type :quit to exit",
                " Please try rating some more movies!! I want to give you a good recommendation and I need your input to do so. Otherwise, type :quit to exit me." 
            ]
            
            response += random.choice(out_of_recommendations_responses)
        
        # 11: Response Variation when user doesn't want recommendations
        elif status == "No More Recs Wanted":
            no_more_recs_wanted_responses = [
                " Ok. Let me know if you change your mind. Ok. Let me know if you change your mind. You can type ':quit' if you are done for now.",
                " Hopefully you have liked the other recs I gave. If not you can keep giving movie inputs for more tailored recommendations",
                " Do you want to keep tailoring your recommendations then? (If so type more movies you liked/disliked in quotes)"
            ]
            
            response += random.choice(no_more_recs_wanted_responses)
        
        # 12: Response Variation when recommending mode response is invalid
        elif status == "Recommending Mode Invalid Input":
            rec_mode_invalid_input_responses = [
                " Sorry. I did not understand your response. Would you like another recommendation? (Please give a more explicit answer like yes or no) or would you like to keep rating moves? (If so, give movies in double quotes)",
                " I didn't get that. Make sure your response includes a yes, no, or a movie in double quotes"
                " Whoops. Your input was not in a format I could process. Please (1) let me know if you want another recommendation, (2) give a new movie opinion, or (3) type :quit to exit",
            ]
            response += random.choice(rec_mode_invalid_input_responses)
        
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        
        return re.findall(r'"([^"]+)"', preprocessed_input)

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        
        # search movie with given title 
        result = self._search_movies(title)
        if result != []:
            return result
        
        system_prompt = """You will respond return 0 if the user input is in English and 1 if it is in a foreign language. Do not include any additional information in your answer."""
        message = f"{title}"
        response = util.simple_llm_call(system_prompt, message, stop=["\n"])
        is_foreign = bool(int(response))

        # if foreign title then we translaate it 
        if is_foreign:   
            translated_title = self._translate_title(title)
            return self._search_movies(translated_title)
        
        return []
    
    ####################### helper functions for find_mvoies_by_title ############################
    def _search_movies(self, title):
        """Helper function for find_movies_by_title 
        Searches for movies in the database that match the given title 
        """

        # Common articles for movie names 
        title_lower = title.lower()
        articles = {"A", "An", "The"}
        result = []

        # check if input contains a 4-digit year in parentheses.
        input_has_year = bool(re.search(r'\(\d{4}\)', title))
                
        with open("data/movies.txt", "r") as file:
            for i, line in enumerate(file):
                # Find movie title within % delimiter
                title_start_idx = line.find('%')
                title_end_idx = line.find('%', title_start_idx + 1)
                
                # If format is incorrect, skip this line
                if title_start_idx == -1 or title_end_idx == -1:
                    continue
                
                # Extract movie title and strip spaces
                curr_movie = line[title_start_idx + 1 : title_end_idx]

                # Check if the common article is at the end and update movie title
                possible_article_idx = curr_movie.rfind(',') #find final comma
                if possible_article_idx != -1: # if comma found
                    possible_article_end_idx = curr_movie.find(' ', possible_article_idx + 2)
                    possible_article = curr_movie[possible_article_idx + 2 : possible_article_end_idx]

                    if possible_article in articles:
                        curr_movie = f"{possible_article} {curr_movie[:possible_article_idx]}{curr_movie[possible_article_end_idx:]}"

                # if input has year, remove any parentheses between text and year 
                if input_has_year:
                    candidate_processed = re.sub(
                        r'\(([^)]*)\)',
                        lambda match: f"({match.group(1)})" if re.fullmatch(r'\d{4}', match.group(1)) else "",
                        curr_movie
                    ).strip()
                    # compare candidate (including year) with the input
                    if title_lower == candidate_processed.lower():
                        result.append(i)

                # if input has no year we remove all contents in parenthesis from the title in database
                else:
                    clean_title = re.sub(r'\([^)]*\)', '', curr_movie).strip()
                    if title_lower == clean_title.lower():
                        result.append(i)

        return result
    
    def _translate_title(self, foreign_title):
        """ Helper function for find_movies_by_title
        Use LLM (prompting for specific output) to translate a movie title from a foreign language to English.
        """
        
        # define system prompt 
        system_prompt = """You are a translation bot who translates movie titles from German, Spanish, French, Danish, and Italian to English.""" +\
        """You will directly translate the movie title to English.""" +\
        """You will solely output the translated title of the movie, with no information. Do not include the original language.""" +\
        """If the year is originally included, then include it in the translation (e.g. 'El Cuaderno (2004)' -> 'The Notebook (2004)')""" +\
        """If the year is not included, only output the translated title (e.g. 'La Guerre du feu' -> 'Quest for Fire')""" +\
        """The output you provide will be compared to titles in a database, so it is VERY important you match the format especifications outlined above."""

        # execute llm call and return response 
        response = util.simple_llm_call(system_prompt, foreign_title, stop=["\n"])

        # remove any additional llm output 
        response = re.sub(r'\((?!\d{4}\))[^)]*\)', '', response)

        # strip and return 
        return response.strip()

    def load_sentiment_dictionary(self, src_filename: str, delimiter: str = ',', header: bool = False):
        """Loads sentiment.txt and stores pre-stemmed words in a dictionary."""
        sentiment_dict = {}
        stemmer = PorterStemmer()

        with open(src_filename, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            if header:
                next(reader)
            for word, sentiment in reader:
                sentiment_dict[stemmer.stem(word)] = sentiment

        return sentiment_dict

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        # Convert input into list of lowercase words
        preprocessed_input = preprocessed_input.lower()
        movie_titles = self.extract_titles(preprocessed_input)

        # Remove movie titles before sentiment analysis
        for title in movie_titles:
            preprocessed_input = preprocessed_input.replace(title.lower(), "")

        # Convert input into list of words
        words = preprocessed_input.split()

        # Initialize sentiment score 
        sentiment = 0

        # Create an instance of the stemmer
        stemmer = PorterStemmer()

        # Setup Negation words
        negation_words = {"not", "never", "no", "n't", "didn't", "doesn't", "isn't", "wasn't", 
                        "shouldn't", "couldn't", "won't", "hadn't", "wouldn't", "don't", "can't"}
        stemmed_negation_words = {stemmer.stem(word) for word in negation_words}

        # Negation Flag 
        Negation = False

        for word in words:
            stemmed_word = stemmer.stem(word)

            # If negation word, do not use it for score and set negation flag
            if stemmed_word in stemmed_negation_words:
                Negation = True
                continue

            if stemmed_word in self.sentiment:
                if self.sentiment[stemmed_word] == 'pos': 
                    # Increase score for pos words or decrease if neg flag set
                    sentiment += -1 if Negation  else 1  
                elif self.sentiment[stemmed_word] == 'neg':
                    # Decrease score for neg words or increase if neg flag set
                    sentiment += 1 if Negation else -1 
                Negation = False  # Reset negation after processing a sentiment word

        # Determine output (1 if score is pos, -1 if score is neg, 0 otherwise)
        if sentiment > 0: 
            return 1
        elif sentiment < 0: 
            return -1
        else: 
            return 0

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        
        # copy original matrix to avoid editing self.ratings
        binarized_ratings = np.copy(ratings)
        
        # Binarize the matrix 
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[(ratings <= threshold) & (ratings > 0)] = -1
        

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        numerator = np.dot(u, v)
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        
        # Avoid division by zero
        if denominator == 0:
            return 0 
        
        similarity = numerator / denominator
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        # Initialize variables: 
        num_movies = ratings_matrix.shape[0]
        predicted_ratings = np.zeros(num_movies)
        
        for i in range(num_movies):
            # skip movies if rated by the user
            if user_ratings[i] != 0:
                continue 
            
            # initialize variables for calculating similarity 
            score = 0.0
                        
            # iterate over every movie the user has rated
            for j in range(num_movies):
                # skip movies if not rated by the user
                if user_ratings[j] == 0:
                    continue
                    
                similarity = self.similarity(ratings_matrix[i],ratings_matrix[j])
                score += similarity * user_ratings[j]
            
            predicted_ratings[i] = score + 1
        
        sorted_indices = np.argsort(predicted_ratings)[::-1]
        recommendations = [i for i in sorted_indices if user_ratings[i] == 0][:k]

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is Movie_Recommender. You are a dedicated movie recommendation chatbot. 
        Your sole purpose is to gather information about a user's movie preferences and provide personalized recommendations after collecting enough data. 
        You follow strict rules to maintain focus and deliver concise, relevant interaction, but make sure you do not ever repeat the exact same response

        Rule #1: You stay on topic and only discuss movies. If a user tries to steer the conversation away from movies, you politely but FIRMLY redirect them back to film-related topics. 
        You do not provide absolutely any information for irrelevant inquiries. You pretend to not know anything not related to movies. 
        Example: 'I can only discuss movies! Anything film-related you'd like to chat about?'

        Rule #2: When a user shares their opinion on a movie, you acknowledge their sentiment and immediately ask about another movie. 
        Example: 'You liked "The Notebook"! What did you think of another movie?'

        Rule #3: You must collect at least 5 movie opinions before providing recommendations. After each valid response, update the counter. If the input does not reference a specific movie, do not increment the counter. 
        Example: 'Got it! That's 3/5 movies. Tell me about another one!' 
        Once they reach five, automatically prompt them for a recommendation request. 
        Example: 'That's 5/5! Would you like a movie recommendation now?'

        Rule #4: You will provide quick, useful, relevant responses. You will avoid unnecessary details, internal thoughts, and parentheticals.

        Rule #5: All movie titles must be enclosed in quotation marks. If a user does not follow this, politely correct them. 
        Example: 'Just to clarify, please use quotation marks for movie titles!'

        Rule #6: If a movie title is ambiguous, you will ask for clarification. 
        Example: 'Did you mean "Dune" (2021) or the original from 1984?'

        Rule #7: Let users choose which movie to discuss. You will not suggest specific movie titles.
        Example: 'What is your opinion on another movie you have watched?'
        Counterexample: 'What did you think of 'Fight Club (1999)'?'""

        Rule #8: give varying responses each time that are associated to the prompt sentiment to each input. Vary each response so it sounds natural and conversational.*
        """

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """

        class EmotionExtractor(BaseModel):
            emotions: list = Field(default_factory=list)
        
        system_prompt = """You are a highly capable, thoughtful, and precise emotion extractor bot. You only label text with emotions from this set: anger, disgust, fear, happiness, sadness, surprise.

        You will follow these guidelines carefully:
            1. anger: Triggered by OBVIOUS feelings of frustration  or hostility. Words like "awful," "bad," or "terrible" usually indicate anger when used in a context of frustration.
            2. disgust: disgust: VERY strong physical revulsion (e.g. “repulsive,” “gross,” or “disgusting”). Use disgust only if the user shows an UNDENIABLE sense of revulsion. Disgust can be thought of as a "yuck" reaction with some of the provided keywords.
            3. fear: Label as fear when the text conveys a sense of being scared, threatened, or alarmed (e.g., "frightening," "terrifying," "horrifying," "startled"). If the text expresses shock or anger without a genuine sense of threat or danger, do not label it as fear.
            4. happiness: Label as happiness when the text shows joy, delight, or being pleased (e.g., "great," "wonderful," "delightful"). Input which states the user simply likes something does not imply happiness. 
            5. sadness: Label as sadness when the text conveys sorrow, grief, or deep depression (e.g., "I am so sad," "I'm heartbroken"). Frustration does not imply sadness unless there is genuine sorrow. Simple dislike also does not imply a sense of sadness. 
            6. surprise: Label as surprise when the text shows an unexpected reaction without the anxiety or threat associated with fear. Surprise may co-occur with other emotions if the reaction is clearly multifaceted.
            7. Joint Emotions: If the input contains clear, distinct signals for multiple emotions (for example, "I was shocked and scared"), list all that are explicitly supported by the context. However, avoid adding extra emotions if only one is strongly indicated
            8. Do NOT include an emotion that isn't clearly supported by the text. For example, expressions like "shockingly bad" should not automatically add fear or disgust if the context clearly indicates anger.

        Output Requirements:
            - Your output must be valid JSON in the exact format: {"emotions": [ ... ]}
            - The only key allowed is "emotions"
            - The value for "emotions" is an array containing zero or more of the allowed emotions in lowercase.
            - If no emotions apply, return an empty array
            - Do not include any additional commentary or extra fields.
        """

        # llm call to get response 
        response = util.json_llm_call(system_prompt, preprocessed_input, EmotionExtractor)

        # extract list of emotions from response 
        if hasattr(response, "emotions"):
            raw_emotions = response.emotions
        elif isinstance(response, dict):
            raw_emotions = response.get("emotions", [])
        else:
            raw_emotions = []

        # return a lower case set 
        emotion_set = {emotion.lower() for emotion in raw_emotions}
        return emotion_set
    
    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """
        if self.llm_enabled: 
            return """This is our LLM-enabled James Bond movie recommender chatbot!"""
        
        return "This is our GUS movie recommender chatbot!"


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
