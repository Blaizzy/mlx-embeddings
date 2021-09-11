# typed: true
# frozen_string_literal: true

require "cask/artifact/abstract_uninstall"

module Cask
  module Artifact
    # Artifact corresponding to the `uninstall` stanza.
    #
    # @api private
    class Uninstall < AbstractUninstall
      def uninstall_phase(**options)
        ORDERED_DIRECTIVES.reject { |directive_sym| directive_sym == :rmdir }
                          .each do |directive_sym|
                            dispatch_uninstall_directive(directive_sym, **options)
                          end
      end

      def post_uninstall_phase(**options)
        dispatch_uninstall_directive(:rmdir, **options)
      end
    end
  end
end
