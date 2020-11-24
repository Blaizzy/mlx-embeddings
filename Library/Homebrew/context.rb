# typed: true
# frozen_string_literal: true

require "monitor"

# Module for querying the current execution context.
#
# @api private
module Context
  extend MonitorMixin

  def self.current=(context)
    synchronize do
      @current = context
    end
  end

  def self.current
    if current_context = Thread.current[:context]
      return current_context
    end

    synchronize do
      @current ||= ContextStruct.new
    end
  end

  # Struct describing the current execution context.
  class ContextStruct
    def initialize(debug: nil, quiet: nil, verbose: nil)
      @debug = debug
      @quiet = quiet
      @verbose = verbose
    end

    def debug?
      @debug == true
    end

    def quiet?
      @quiet == true
    end

    def verbose?
      @verbose == true
    end
  end

  def debug?
    Context.current.debug?
  end

  def quiet?
    Context.current.quiet?
  end

  def verbose?
    Context.current.verbose?
  end

  def with_context(**options)
    old_context = Thread.current[:context]

    new_context = ContextStruct.new(
      debug:   options.fetch(:debug, old_context&.debug?),
      quiet:   options.fetch(:quiet, old_context&.quiet?),
      verbose: options.fetch(:verbose, old_context&.verbose?),
    )

    Thread.current[:context] = new_context

    yield
  ensure
    Thread.current[:context] = old_context
  end
end
