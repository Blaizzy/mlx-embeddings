# typed: false
# frozen_string_literal: true

module Utils
  # Wrappers for Ruby core methods that accept splat arguments. This file is `typed: false` by design, but allows
  # other files to enable typing while making use of the wrapped methods.
  #
  # @api private
  module Splat
    extend T::Sig

    # Wrapper around `Process.kill` that accepts an array of pids.
    # @see https://ruby-doc.org/3.2.1/Process.html#method-c-kill Process.kill
    # @see https://github.com/sorbet/sorbet/blob/eaebdcd/rbi/core/process.rbi#L793-L800 Sorbet RBI for `Process.kill`
    sig { params(signal: T.any(Integer, Symbol, String), pids: T::Array[Integer]).returns(Integer) }
    def self.process_kill(signal, pids)
      Process.kill(signal, *pids)
    end
  end
end
