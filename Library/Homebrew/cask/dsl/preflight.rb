# typed: strict
# frozen_string_literal: true

module Cask
  class DSL
    # Class corresponding to the `preflight` stanza.
    #
    # @api private
    class Preflight < Base
      include Staged
    end
  end
end
