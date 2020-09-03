# frozen_string_literal: true

module Language
  module Python
    class << self
      module Compat
        def rewrite_python_shebang(_python_path)
          odisabled "Language::Python.rewrite_python_shebang",
                    "Utils::Shebang.rewrite_shebang and Shebang.python_shebang_rewrite_info(python_path)"
        end
      end

      prepend Compat
    end
  end
end
