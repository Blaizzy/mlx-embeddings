# frozen_string_literal: true

require "forwardable"
require "uri"

module RuboCop
  module Cop
    module Cask
      # This cop checks that a cask's homepage ends with a slash
      # if it does not have a path component.
      class HomepageUrlTrailingSlash < Cop
        include OnHomepageStanza

        MSG_NO_SLASH = "'%<url>s' must have a slash after the domain."

        def on_homepage_stanza(stanza)
          url_node = stanza.stanza_node.first_argument

          url = if url_node.dstr_type?
            # Remove quotes from interpolated string.
            url_node.source[1..-2]
          else
            url_node.str_content
          end

          return unless url&.match?(%r{^.+://[^/]+$})

          add_offense(url_node, location: :expression,
                                message:  format(MSG_NO_SLASH, url: url))
        end

        def autocorrect(node)
          domain = URI(node.str_content).host

          # This also takes URLs like 'https://example.org?path'
          # and 'https://example.org#path' into account.
          corrected_source = node.source.sub("://#{domain}", "://#{domain}/")

          lambda do |corrector|
            corrector.replace(node.source_range, corrected_source)
          end
        end
      end
    end
  end
end
