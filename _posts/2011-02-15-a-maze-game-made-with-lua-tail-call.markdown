---
layout: post
type: post
title: A maze game made with Lua's tail-call
tags: lua tail-recursive programing
description: A maze game, designed using a state machine. In Lua, tail-calls as shown below are GOTOs, which doesn't create a stack. Infinite tail-call are permitted as not to raise stackoverflow. In Python, however, this will be a problem.
---
A maze game, designed using a state machine. In Lua, tail-calls as shown below are GOTOs, which doesn't create a stack. Infinite tail-call are permitted as not to raise stackoverflow. In Python, however, this will be a problem.

```lua

function room1()
    local move = io.read()
    if move == "south" then return room3()
    elseif move == "east" then return room2()
    else
        print("invalid move")
        return room1() -- stay in the same room
    end 
end

function room2()
    local move = io.read()
    if move == "south" then return room4()
    elseif move == "west" then return room1()
    else
        print("invalid move")
        return room2()
    end 
end

function room3()
    local move = io.read()
    if move == "north" then return room1()
    elseif move == "east" then return room4()
    else
        print("invalid move")
        return room3()
    end 
end

function room4()
    print("Congrats!")
end

room1()

```