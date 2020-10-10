# typed: false
# frozen_string_literal: true

require "cask/artifact/abstract_uninstall"

module Cask
  module Artifact
    # Artifact corresponding to the `zap` stanza.
    #
    # @api private
    class Zap < AbstractUninstall
      def zap_phase(**options)
        dispatch_uninstall_directives(**options)
      end
    end
  end
end
