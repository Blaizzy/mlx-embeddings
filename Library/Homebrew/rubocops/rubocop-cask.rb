# typed: strict
# frozen_string_literal: true

require "rubocop"

require_relative "cask/constants/stanza"

require_relative "cask/ast/stanza"
require_relative "cask/ast/cask_header"
require_relative "cask/ast/cask_block"
require_relative "cask/extend/string"
require_relative "cask/extend/node"
require_relative "cask/mixin/cask_help"
require_relative "cask/mixin/on_homepage_stanza"
require_relative "cask/desc"
require_relative "cask/homepage_url_trailing_slash"
require_relative "cask/no_dsl_version"
require_relative "cask/stanza_order"
require_relative "cask/stanza_grouping"
