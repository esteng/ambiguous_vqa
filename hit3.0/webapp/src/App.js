import logo from './logo.svg';
import './App.css';
import React, { Component } from 'react';
import mock from './mock.json';
import Header from './Header'
import ImagePanel from './ImagePanel'
import GroupPanel from './GroupPanel'
// import QuestionPanel from './QuestionPanel'
// import ButtonPanel from './ButtonPanel'

// import Button from "@material-ui/core/Button/Button";
// Main class that serves the app  
class App extends Component {
  // needs to
  // 1. display image and question
  // 2. display answer groups (pre-initialized)
  // 3. allow drag and drop groups 
  // 4. display text box per group 
  // 5. have a submit button
  // 6. have a skip button with a required text box  
  constructor(props){
    super(props);
    // initalize
    var el = document.querySelector("#hiddenSubmit");
    if (el) {
      // if el is not null, than this code is loaded by Turkle.
      el.style.display = 'none';
      // this.initializeStates(window.imgUrl, window.questionStr, window.answerGroups, window.answerQuestions);
      const {imgUrl, questionStr, answerGroups, answerQuestions} = {"imageUrl": window.imgUrl, "questionStr": window.questionStr, "answerGroups": window.answerGroups, "answerQuestions": window.answerQuestions};
      this.state = {
        imgUrl: imgUrl,
        questionStr: questionStr,
        answerGroups: answerGroups,
        answerQuestions: answerQuestions
      };
      // this.state = {imgUrl: window.imgUrl}
    } 
    else {
      // this code is executeed independently, not inside Turkle.
      const {imgUrl, questionStr, answerGroups, answerQuestions} = mock;
      this.state = {
        imgUrl: imgUrl,
        questionStr: questionStr,
        answerGroups: answerGroups,
        answerQuestions: answerQuestions
      };
    }
  }
 
  handleChange = event => {
    this.setState({"questionStr": event.target.value})
    console.log(this.state)
  }

  render() {

    return (
    <div>
      <Header trainingBtn={null} handleFontSizeChange={false}/>
      <ImagePanel imgUrl={this.state.imgUrl} questionStr={this.state.questionStr} /> 
      <GroupPanel answerGroups={this.state.answerGroups} answerQuestions={this.state.answerQuestions}/>
      <input className="submit" onClick={this.handleSubmit}
               type="submit" value="Submit" />
      </div>        
    );
  }
}



export default App;
