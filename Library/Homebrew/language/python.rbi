# typed: strict

module Language::Python
  module Shebang
    include Kernel
  end

  module Virtualenv
    requires_ancestor { Formula }
  end
end
