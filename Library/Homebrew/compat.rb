# frozen_string_literal: true

require "compat/dependencies_helpers"
require "compat/cli/parser"
require "compat/extend/nil"
require "compat/extend/string"
require "compat/formula"
require "compat/language/java"
require "compat/language/python"
require "compat/os/mac" if OS.mac?
