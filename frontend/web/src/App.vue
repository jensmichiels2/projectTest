<template>
  <div id="app" class="">
    <h1 class="text-center mt-4">ZEEHOND</h1>
    <div class="row mt-4" style="height: calc(100vh - 100px);">
      <div class="jira-container col-3">
        <h2>Jira Tickets</h2>
        <ul class="list-group">
          <li v-for="ticket in jiraTickets" :key="ticket.key" class="list-group-item">
            <strong>{{ ticket.key }}</strong>: {{ ticket.summary }}
          </li>
        </ul>
      </div>
      <div class="chat-container col-9">
        <div class="chat-box">
          <div v-for="(msg, index) in messages" :key="index" class="message" :class="{ 'user-message': msg.fromUser, 'bot-message': !msg.fromUser }">
            <p>{{ msg.text }}</p>
          </div>
        </div>
        <div class="input-container">
          <input v-model="question" @keyup.enter="askChatGPT" class="form-control" placeholder="Type your question..." />
          <button @click="askChatGPT" class="btn btn-primary mt-2">Send</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      question: '',
      messages: [],
      jiraTickets: [] // Initialize the jiraTickets array
    };
  },
  methods: {
    async askChatGPT() {
      if (!this.question) return;
      this.messages.push({ text: this.question, fromUser: true });
      try {
        const response = await axios.post('http://localhost:8000/ask', { question: this.question });
        this.messages.push({ text: response.data.answer, fromUser: false });
        this.question = ''; // Clear input after sending
      } catch (error) {
        console.error("Error asking ChatGPT:", error);
      }
    },
    async fetchJiraTickets() {
      try {
        const response = await axios.get('http://localhost:8000/jira-tickets'); // Use the new endpoint
        this.jiraTickets = response.data.tickets || [];  // Update the tickets data
      } catch (error) {
        console.error("Error fetching Jira tickets:", error);
      }
    }
  },
  mounted() {
    this.fetchJiraTickets(); // Fetch tickets when the component is mounted
  }
};
</script>

<style>
#app {
  margin-top: 20px;
}

.jira-container {
  height: calc(100vh - 100px); /* Full height minus header */
  overflow-y: auto;
  border-right: 1px solid #ced4da;
  background-color: #f8f9fa;
}

.chat-container {
  height: calc(100vh - 100px); /* Full height minus header */
  width: 1000px;
  overflow-y: auto;
  padding: 20px;
  background-color: #f8f9fa;
}

.chat-box {
  margin-bottom: 20px;
}

.message {
  padding: 10px;
  margin: 5px 0;
  border-radius: 5px;
}

.user-message {
  background-color: #d1ecf1;
  text-align: right;
}

.bot-message {
  background-color: #c3e6cb;
  text-align: left;
}

.input-container {
  display: flex;
  flex-direction: column;
}

input {
  width: 100%;
}

.list-group-item {
  cursor: pointer;
}

.list-group-item:hover {
  background-color: #e2e6ea;
}
</style>




