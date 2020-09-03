# frozen_string_literal: true

module Language
  module Haskell
    module Cabal
      module Compat
        def cabal_sandbox(_options = {})
          odisabled "Language::Haskell::Cabal.cabal_sandbox"
        end

        def cabal_sandbox_add_source(*_args)
          odisabled "Language::Haskell::Cabal.cabal_sandbox_add_source"
        end

        def cabal_install(*_args)
          odisabled "Language::Haskell::Cabal.cabal_install",
                    "cabal v2-install directly with std_cabal_v2_args"
        end

        def cabal_configure(_flags)
          odisabled "Language::Haskell::Cabal.cabal_configure"
        end

        def cabal_install_tools(*_tools)
          odisabled "Language::Haskell::Cabal.cabal_install_tools"
        end

        def install_cabal_package(*_args, **_options)
          odisabled "Language::Haskell::Cabal.install_cabal_package",
                    "cabal v2-update directly followed by v2-install with std_cabal_v2_args"
        end
      end

      prepend Compat
    end
  end
end
