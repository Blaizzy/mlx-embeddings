# frozen_string_literal: true

require "rubocop"

require_relative "rubocop/sorbet"
require_relative "rubocop/sorbet/version"
require_relative "rubocop/sorbet/inject"

RuboCop::Sorbet::Inject.defaults!

require_relative "rubocop/cop/sorbet_cops"
