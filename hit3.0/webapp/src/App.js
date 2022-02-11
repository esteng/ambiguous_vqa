import './App.css';
import React, { Component } from 'react';
import mock from './mock.json';
import Header from './Header'
import ImagePanel from './ImagePanel'
import GroupPanel from './GroupPanel'
import {Alert, FormControlLabel, Checkbox, Button, TextField } from '@mui/material';


// import QuestionPanel from './QuestionPanel'
// import ButtonPanel from './ButtonPanel'

// import Button from "@material-ui/core/Button/Button";
// Main class that serves the app  

const reorder = (list, startIndex, endIndex) => {
  const result = Array.from(list);
  const [removed] = result.splice(startIndex, 1);
  result.splice(endIndex, 0, removed);

  return result;
};

/**
 * Moves an item from one list to another list.
 */
const move = (source, destination, droppableSource, droppableDestination) => {
  const sourceClone = Array.from(source);
  const destClone = Array.from(destination);
  const [removed] = sourceClone.splice(droppableSource.index, 1);

  destClone.splice(droppableDestination.index, 0, removed);

  const result = {};
  result[droppableSource.droppableId] = sourceClone;
  result[droppableDestination.droppableId] = destClone;

  return result;
};

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
      // hide the other inputs as well 
      var answerGroupsInput = document.querySelector("#answerGroupsInput")
      answerGroupsInput.style.display = "none"
      var answerQuestionsInput = document.querySelector("#answerQuestionsInput")
      answerQuestionsInput.style.display = "none"
      var isSkipInput = document.querySelector("#isSkipInput")
      isSkipInput.style.display = "none"
      var skipReasonInput = document.querySelector("#skipReasonInput")
      skipReasonInput.style.display = "none"
      


      const {imgUrl, questionStr, answerGroups, answerQuestions} = {"imgUrl": window.imgUrl, "questionStr": window.questionStr, "answerGroups": window.answerGroups, "answerQuestions": window.answerQuestions};


      this.state = {
        imgUrl: imgUrl,
        questionStr: questionStr,
        answerGroups: answerGroups,
        answerQuestions: answerQuestions,
        isSkipped: false,
        skipReason: "All answer the same question",
        errorMessage: null,
        isOverflowing: false,
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
        answerQuestions: answerQuestions,
        isSkipped: false,
        skipReason: "All answer the same question",
        errorMessage: null,
        isOverflowing: false,
      };
    }
    this.originalState = JSON.parse(JSON.stringify(this.state))

  }
 
  handleChange = event => {
    this.setState({"questionStr": event.target.value})
  }

  handleSubmit = event => {
    console.log(this.state.answerQuestions)
    var n = document.querySelector("#answerGroupsInput")
    if (n) {
      n.value = JSON.stringify(this.state.answerGroups)
      var n2 = document.querySelector("#answerGroupsInput")
    }
    var n = document.querySelector("#answerQuestionsInput")
    if (n) {
      n.value = JSON.stringify(this.state.answerQuestions)
    }
    var n = document.querySelector("#isSkipInput")
    if (n) {
      n.value = JSON.stringify(this.state.isSkipped)
    }
    var n = document.querySelector("#skipReasonInput")
    if (n) {
      n.value = JSON.stringify(this.state.skipReason)
    }
    // }


  }
  handleSkip = event => {
    // set skip 
    this.setState({isSkipped: !this.state.isSkipped})
  }
  handleSkipReason = event => {
    this.setState({skipReason: event.target.value})
    if (event.target.value.length > 0){
      this.setState({errorMessage: null})
    }
  }

  handleAnswerGroupUpdate = event => {
    this.setState({answerGroups: event.target.value})
  }

  handleAnswerQuestionUpdate = (event, index) => {
    var event;
    // this is currently fucked, questions only have one value, need to have indexed values 
    var new_answerQuestions = this.state.answerQuestions;
    new_answerQuestions[index] = event.target.value;
    this.setState({answerQuestions: new_answerQuestions});
  }

  handleDelete = (ind, index) => {
    const newStateItems = [...this.state.answerGroups];
    const newStateQuestions = [...this.state.answerQuestions];
    newStateItems[ind].splice(index, 1);
    if (newStateItems[ind].length === 0) {
      newStateQuestions.splice(ind, 1);
    }
    this.setState({answerGroups: newStateItems.filter(group => group.length)});
    this.setState({answerQuestions: newStateQuestions.filter(group => group.length)});
    // this.setState({answerQuestions: newStateQuestions})
    if (this.state.answerGroups.length > 2) { 
      this.setState({isOverflowing: true})
    } else {
      this.setState({isOverflowing: false})
    }
  }

  handleAdd = () => {
    this.setState({answerGroups: [...this.state.answerGroups, []]})
    this.setState({answerQuestions: [...this.state.answerQuestions, ""]})
    if (this.state.answerGroups.length > 2) { 
      this.setState({isOverflowing: true})
    } else {
      this.setState({isOverflowing: false})
    }
  }

  handleReset = () => {
    // this.setState({...this.originalState}) 
    // this.setState({isSkipped: false})
    window.location.reload(false)
  }

  onDragEnd = result => {
    const { source, destination } = result;
    // dropped outside the list
    if (!destination) {
      return;
    }
    const sInd = +source.droppableId;
    const dInd = +destination.droppableId;

    var itemsToSet = null;
    var questionsToSet = this.state.answerQuestions; 
    if (sInd === dInd) {
      const items = reorder(this.state.answerGroups[sInd], source.index, destination.index);
      const newStateItems = [...this.state.answerGroups];
      newStateItems[sInd] = items;
      itemsToSet = newStateItems;
    } else {
      const result = move(this.state.answerGroups[sInd], this.state.answerGroups[dInd], source, destination);
      const newStateItems = [...this.state.answerGroups];
      newStateItems[sInd] = result[sInd];
      newStateItems[dInd] = result[dInd];
      itemsToSet = newStateItems.filter(group => group.length)
      if (itemsToSet[sInd].length === 0) {
        questionsToSet = questionsToSet.splice(sInd, 1)
        this.setState({answerQuestions: questionsToSet})
      }
    }
    this.setState({answerGroups: itemsToSet})
  }

  render() {
    var skipReasonBox;
    if (this.state.isSkipped) {

      skipReasonBox = <TextField 
                            id="skip-reason" 
                            required helperText="Reason for skipping" 
                            variant="outlined" 
                            fullWidth 
                            margin="dense"
                            defaultValue={"All answers to the same question"}
                            onChange={this.handleSkipReason}/>
    }
    else {
      skipReasonBox = ""
    }

    return (
    <div>
      <Header trainingBtn={null} handleFontSizeChange={false}/>
      <div className='containter flex-direction'>
        <div style={{width: "500px",  float: "left"}}>
          <ImagePanel imgUrl={this.state.imgUrl} questionStr={this.state.questionStr} /> 
        </div>
        <div style={{width: "1000px" ? this.state.isOverflowing : "500px", float: "left"}}>
          <div>
            <GroupPanel answerGroups={this.state.answerGroups} 
                        answerQuestions={this.state.answerQuestions} 
                        dragDropHandler={this.onDragEnd} 
                        questionHandler={this.handleAnswerQuestionUpdate}
                        addHandler={this.handleAdd}
                        resetHandler={this.handleReset}
                        deleteHandler={this.handleDelete}/> 
          </div>
          <div>
            <div style={{width: "50%", float:"left"}}>
              <FormControlLabel control={<Checkbox name="skipCheck" value="value" onClick={this.handleSkip}/>} label="skip"/>
              {skipReasonBox}
            </div>
            <div style={{width: "50%", float:"left"}}>
              <input  onClick={this.handleSubmit} className="button1"
                type="submit" value="Submit" />
            </div>
          </div>
        </div>
      </div>
    </div>
        

        

    );
  }
}



export default App;
