# frozen_string_literal: true

module Cask
  class DSL
    class Version < ::String
      module Compat
        def dots_to_slashes
          odeprecated "#dots_to_slashes"
          version { tr(".", "/") }
        end

        def hyphens_to_slashes
          odeprecated "#hyphens_to_slashes"
          version { tr("-", "/") }
        end

        def underscores_to_slashes
          odeprecated "#underscores_to_slashes"
          version { tr("_", "/") }
        end
      end

      prepend Compat
    end
  end
end
