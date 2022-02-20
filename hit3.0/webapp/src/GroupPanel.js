import { Button, TextField, Typography } from "@material-ui/core";
import React, { Component } from "react";
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import Alert from '@mui/material/Alert';
import "./GroupPanel.css"


// fake data generator

const grid = 8;

const getItemStyle = (isDragging, draggableStyle) => ({
  // some basic styles to make the items look a bit nicer
  userSelect: "none",
  padding: grid * 2,
  margin: `0 0 ${grid}px 0`,

  // change background colour if dragging
  background: isDragging ? "lightgreen" : "grey",
  borderRadius: "15px",

  // styles we need to apply on draggables
  ...draggableStyle
});
const getListStyle = isDraggingOver => ({
  background: isDraggingOver ? "lightblue" : "lightgrey",
  padding: grid,
  width: 250,
  borderRadius: "15px",
});

class GroupPanel extends Component {
    constructor(props) {
      super(props);
      this.state = {
        answerGroups: props.answerGroups,
        answerQuestions: props.answerQuestions,
        fieldIds: [...props.answerGroups.keys()].map(function (x) {return x.toString();} ),
        items: props.answerGroups,
        errorMessage: props.errorMessage,
      };

      this.dragDropHandler = props.dragDropHandler
      this.questionHandler = props.questionHandler
      this.deleteHandler = props.deleteHandler
      this.addHandler = props.addHandler
      this.resetHandler = props.resetHandler

    }

    // componentWillReceiveProps(nextProps) {
      // this.setState({ answerGroups: nextProps.answerGroups, 
                      // answerQuestions: nextProps.answerQuestions});  
    // } 
    static getDerivedStateFromProps(nextProps, prevState){
      return {answerGroups: nextProps.answerGroups,
             answerQuestions: nextProps.answerQuestions, 
             fieldIds: [...nextProps.answerGroups.keys()].map(function (x) {return x.toString();} ),
             items: nextProps.answerGroups,
             errorMessage: nextProps.errorMessage,
            }
    }


    render(){
        return (
            <div>
              <div>
                <Button
                    variant="contained"
                    // onClick={() => {
                    // this.setState({items: [...this.state.items, []]});
                    // }}
                    onClick={() => this.addHandler()}>
                    Add answer group
                </Button>
                <Button
                    variant="contained"
                    // onClick={() => {
                    // this.setState({items: [...this.state.items, []]});
                    // }}
                    onClick={() => this.resetHandler()}>
                    Reset
                </Button>
              </div>
              <div style={{ display: "flex" }}>
                <DragDropContext onDragEnd={this.dragDropHandler}>
                  {this.state.items.map((el, ind) => (
                    <div>
                      {/* */}
                      
                        <Typography variant="h6">Q{ind}:
                            <TextField 
                                id={this.state.fieldIds[ind]} 
                                helperText="Edit Question" 
                                variant="outlined" 
                                // defaultValue={this.state.answerQuestions[ind]} 
                                error
                                helperText={this.state.errorMessage[ind] ? this.state.errorMessage[ind] : null}
                                value={this.state.answerQuestions[ind] || ''}
                                fullWidth
                                margin="dense"
                                onChange={(event) => this.questionHandler(event, ind)}
                            />
                            </Typography>
                        <Droppable key={ind} droppableId={`${ind}`}>
                        {(provided, snapshot) => (
                            <div
                            ref={provided.innerRef}
                            style={getListStyle(snapshot.isDraggingOver)}
                            {...provided.droppableProps}
                            >
                            {el.map((item, index) => (
                                <Draggable
                                key={item.id}
                                draggableId={item.id}
                                index={index}
                                >
                                {(provided, snapshot) => (
                                    <div
                                    ref={provided.innerRef}
                                    {...provided.draggableProps}
                                    {...provided.dragHandleProps}
                                    style={getItemStyle(
                                        snapshot.isDragging,
                                        provided.draggableProps.style
                                    )}
                                    >
                                    <div
                                        style={{
                                        display: "flex",
                                        justifyContent: "space-around"
                                        }}
                                    >
                                        {item.content}
                                        <Button variant="contained" 
                                        onClick={() => this.deleteHandler(ind, index) }>
                                        delete
                                        </Button>
                                    </div>
                                    </div>
                                )}
                                </Draggable>
                            ))}
                            {provided.placeholder}
                            </div>
                        )}
                        </Droppable>
                    </div>
                  ))}
                </DragDropContext>
              </div>
            </div>
          );
    }
    
}
export default GroupPanel;