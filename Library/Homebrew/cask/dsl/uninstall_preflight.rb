# typed: strict
# frozen_string_literal: true

require "cask/staged"

module Cask
  class DSL
    # Class corresponding to the `uninstall_preflight` stanza.
    #
    # @api private
    class UninstallPreflight < Base
      include Staged
    end
  end
end
