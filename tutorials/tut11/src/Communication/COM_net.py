from queue import Queue

"""
COM_net
    implements communication module
    act as a post-office/com-network according to possible com. settings
    
    also include Message class - general type of a message between agents (can include any data
"""



class Message:
    """
    Class Message
        holds:
            - data: the essence of the message
            - author: author(sender) - default None
            - time_stamp: time of sending (discrete or continues) - default None
            - p2p_target: the target of the message - for p2p network (not implemented)
    """
    def __init__(self, data, author = None, time_stamp = None, p2p_target = None):
        self.data = data
        self.author = author
        self.time_stamp = time_stamp
        self.size = self.data.__sizeof__()           #len(data)



class COM_net:
    """
    class COM_net
        implements communication module
        act as a post-office/com-network according to possible com. settings

        input:
            - Delivery_target_type : str
                "All" - send the Messages of each agent to all of the agents - defualt is All
                ... otherwise -> as 'written' in the target of each message / sent to all Teammates
                ... TODO - implement broadcast/other option
            - Team group - teammates to recieve message of others
            - Delivert_latency - holds the message X time-steps before delievery, defualt - no latency
            - Noise_Net - noise function for the network

    """
    def __init__(self, Delivery_target_type : str = "All", Delivery_latency : int = 0 , Noise_Net = None, Team_group = None ):
        self.Delivery_target_type = Delivery_target_type

        self.Delivert_latency = Delivery_latency
        self.delivery_buffer = Queue()
        for i in range(Delivery_latency):
            self.delivery_buffer.put(None)

        self.Noise_net = Noise_Net
        self.Team = Team_group

    """
    update_delievery 
        calculates agents delievery messages to be recieved at current time-step according to communication model
        
        input
            - (in-)Joint_messages - dict of messages {agent : agent's Message} that was sent in this time-step
            - Agents - name of agents that take part of the next timestep delievery
            
        output 
            - dict of next step {agent-to-recieve : joint_messages for that agent}
    """
    def update_delievery(self, Joint_messages : list, Agents):
        self.delivery_buffer.put(Joint_messages)
        Out_Joint_message = self.delivery_buffer.get()
        if self.Delivery_target_type == "All":
            # all_messages = [message.value for message in Out_Joint_message]
            return {agent : Out_Joint_message for agent in Agents}
        elif self.Team != None:
            out_message_dict = {m.author: m for m in Out_Joint_message}
            Team_messages = {agent : out_message_dict[agent] for agent in self.Team}
            all_messages = [message for message in Team_messages]
            return {agent : all_messages for agent in Agents}
        # TODO - add other com methods (broadcast, p2p etc), add and use Noise_net function
        return {}



