# typed: strict
# frozen_string_literal: true

require "sorbet-runtime"
require "stringio"

module RBI
  class Error < StandardError; end
end

require "rbi/loc"
require "rbi/model"
require "rbi/visitor"
require "rbi/index"
require "rbi/rewriters/add_sig_templates"
require "rbi/rewriters/merge_trees"
require "rbi/rewriters/nest_singleton_methods"
require "rbi/rewriters/nest_non_public_methods"
require "rbi/rewriters/group_nodes"
require "rbi/rewriters/sort_nodes"
require "rbi/parser"
require "rbi/printer"
require "rbi/version"
