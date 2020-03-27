# frozen_string_literal: true

module Language
  module Python
    class << self
      module Compat
        def rewrite_python_shebang(python_path)
          Pathname.pwd.find do |f|
            Utils::Shebang.rewrite_shebang(Shebang.python_shebang_rewrite_info(python_path), f)
          end
        end
      end

      prepend Compat
    end
  end
end
