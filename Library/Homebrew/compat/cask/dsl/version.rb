# frozen_string_literal: true

module Cask
  class DSL
    class Version < ::String
      module Compat
        def dots_to_slashes
          odisabled "#dots_to_slashes"
        end

        def hyphens_to_slashes
          odisabled "#hyphens_to_slashes"
        end

        def underscores_to_slashes
          odisabled "#underscores_to_slashes"
        end
      end

      prepend Compat
    end
  end
end
